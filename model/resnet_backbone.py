from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import  transforms
import pickle
import os.path
import datetime
import numpy as np


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.Adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear   = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.Adapool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_unlabel(nn.Module):
    def __init__(self, block, num_blocks, nclass_label=5, nclass_unlabel=5):
        super(ResNet_unlabel, self).__init__()
        self.in_planes = 64
        self.Adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.in_planes = 256
        self.layer_unlabel = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear   = nn.Linear(512*block.expansion, nclass_label)
        self.linear_pca = nn.Linear(512 * block.expansion, nclass_unlabel)
        self.linear_bce = nn.Linear(512 * block.expansion, nclass_unlabel)
        self.center = nn.Parameter(torch.Tensor(nclass_unlabel, nclass_unlabel))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, flag=0):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_label = self.layer4(x)
        feat_label = self.Adapool(feat_label)
        feat_label = feat_label.view(feat_label.size(0), -1)
        if flag==0:
            out_label = self.linear(feat_label)
            return feat_label, out_label
        elif flag==1:
            feat_unlabel = self.layer_unlabel(x)
            feat_unlabel = F.avg_pool2d(feat_unlabel, 4)
            feat_unlabel = feat_unlabel.view(feat_unlabel.size(0), -1)
            feat = feat_label + feat_unlabel
            out_unlabel = self.linear_pca(feat)
            return feat_unlabel, out_unlabel
        elif flag==2:
            feat_unlabel = self.layer_unlabel(x)
            feat_unlabel = self.Adapool(feat_unlabel)
            feat_unlabel = feat_unlabel.view(feat_unlabel.size(0), -1)
            feat = feat_label + feat_unlabel
            out_label = self.linear(feat_label)
            out_pca = self.linear_pca(feat)
            out_bce = self.linear_bce(feat_unlabel)

            return feat_label, out_label, out_pca, out_bce


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion*planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut,torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)],1)
        else:
            out += self.shortcut(x)

        out = F.relu(out)
        return out


