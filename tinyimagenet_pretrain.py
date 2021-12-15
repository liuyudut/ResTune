import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch
from models.resnet_3x3 import ResNet, BasicBlock
from data.tinyimagenet_loader import TinyImageNetLoader
from tqdm import tqdm
import numpy as np
import os


def train(model, train_loader, labeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label) in enumerate(tqdm(train_loader)):
            x, label = x.to(args.device), label.to(args.device)
            output = model(x)
            loss = criterion1(output, label)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        eval(model, labeled_eval_loader, args)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.model_dir)
            print("model saved to {}.".format(args.model_dir))


def eval(model, test_loader, args):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    for batch_idx, (x, label) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)
        output = model(x)
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets,
                                                                                                              preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return preds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--milestones', default=[40, 60, 80], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=20, type=int)
    parser.add_argument('--num_labeled_classes', default=180, type=int)
    parser.add_argument('--model_name', type=str, default='tinyimagenet_labeled_classes_pretraining')
    parser.add_argument('--dataset_root', type=str, default='../Datasets/TinyImageNet/')
    parser.add_argument('--exp_root', type=str, default='./results/')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = 'pretrained' #os.path.basename(__file__).split(".")[0]
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'_{}_New.pth'.format(args.num_labeled_classes)

    labeled_train_loader = TinyImageNetLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
    labeled_test_loader = TinyImageNetLoader(root=args.dataset_root, batch_size=args.batch_size, split='val_folders', aug=None, shuffle=False, target_list=range(args.num_labeled_classes))

    model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes).to(args.device)
    train(model, labeled_train_loader, labeled_test_loader, args)
    eval(model, labeled_test_loader, args)

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

