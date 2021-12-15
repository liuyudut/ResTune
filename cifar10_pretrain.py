import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter, accuracy, seed_torch
from data.cifarloader import CIFAR10Loader
#from models.vgg import VGG_net
from models.resnet_backbone import ResNet, BasicBlock
import os
import numpy as np
import sys,time


def train(model, train_loader, eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion=nn.CrossEntropyLoss().cuda(args.device)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        entropy_loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, _) in enumerate(train_loader):
            x, target = x.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            output= model(x)

            loss = criterion(output, target)
            acc = accuracy(output, target)
            loss.backward()
            optimizer.step()
            acc_record.update(acc[0].item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))
        print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
        eval(model, eval_loader, args)

        if epoch%50==0:
            torch.save(model.state_dict(), args.model_dir)
            print("model saved to {}.".format(args.model_dir))

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def eval(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
    for batch_idx, (x, label, _) in enumerate(test_loader):
        x, target = x.to(args.device), label.to(args.device)
        output = model(x)
        acc = accuracy(output, target)
        acc_record.update(acc[0].item(), x.size(0))
    print('Test: Avg Acc: {:.4f}'.format(acc_record.avg))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--milestones', default=[20, 40, 60, 80], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='cifar10_labeled_classes_pretraining')
    parser.add_argument('--dataset_root', type=str, default='../Datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'_{}.pth'.format(args.num_labeled_classes)
    
    labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
    labeled_test_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(args.num_labeled_classes))
    
    ## Net
    model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes).to(args.device)

    ## Train 
    train(model, labeled_train_loader, labeled_test_loader, args)

    ## Test 
    eval(model, labeled_test_loader, args)

    ## Save 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))



