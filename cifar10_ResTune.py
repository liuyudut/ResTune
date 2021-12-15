import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD, lr_scheduler
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, str2bool
from utils import ramps
#from models.vgg_RCL import VGG_net
from models.resnet_backbone import ResNet, BasicBlock, ResNet_unlabel
from modules.module import feat2prob, target_distribution
from data.cifarloader import CIFAR10Loader
from tqdm import tqdm
import numpy as np
import warnings
import os
import scipy.io as sio
import copy
from copy import deepcopy
import torch.nn.functional as func
import time

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, neg=True, batch=False):
        b = self.softmax(x) * self.logsoft(x)
        if batch:
            return -1.0 * b.sum(1)
        if neg:
            return -1.0 * b.sum()/x.size(0)
        else:
            return b.sum()/x.size(0)

def init_prob_kmeans(model, eval_loader, args):
    torch.manual_seed(args.seed)
    #model = model.to(args.device)
    # cluster parameter initiate
    model.eval()
    targets = np.zeros(len(eval_loader.dataset))
    feats = np.zeros((len(eval_loader.dataset), 512))
    with torch.no_grad():
        for _, (x, label, idx) in enumerate(eval_loader):
            x = x.to(args.device)
            feat_label = model(x)
            idx = idx.data.cpu().numpy()
            feats[idx, :] = feat_label.data.cpu().numpy()
            targets[idx] = label.data.cpu().numpy()
    # evaluate clustering performance
    pca = PCA(n_components=args.n_clusters)
    feats = pca.fit_transform(feats)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(feats)
    
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs


def compute_sim_loss(feat_old, feat_new, sim_criterion, args):
    feat_old = func.normalize(feat_old, p=2, dim=1)
    feat_new = func.normalize(feat_new, p=2, dim=1)
    one_batch = torch.ones(feat_new.size(0)).to(args.device)
    sim_loss = sim_criterion(feat_old, feat_new, one_batch)
    return sim_loss


def train(old_model, new_model, train_loader, eva_loader, args):
    mse_criterion = torch.nn.MSELoss().to(args.device)
    bce_criterion = torch.nn.BCEWithLogitsLoss().to(args.device)
    entropy_criterion = HLoss().to(args.device)
    bce_criterion = BCE().to(args.device)
    optimizer = SGD(new_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        total_loss_record = AverageMeter()
        cluster_loss_record = AverageMeter()
        rank_loss_record = AverageMeter()
        reg_loss_record = AverageMeter()
        new_model.train()
        old_model.eval()
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(args.device), x_bar.to(args.device)
            feat_label, out_label, out_pca, out_bce = new_model(x, flag=2)
            _, _, out_pca_bar, _ = new_model(x_bar, flag=2)
            prob_pca = feat2prob(out_pca, new_model.center)
            prob_pca_bar = feat2prob(out_pca_bar, new_model.center)
            prob = F.softmax(out_bce, dim=1)
            sharp_loss = F.kl_div(prob_pca.log(), args.p_targets[idx].float().to(args.device))

            consistency_loss = F.mse_loss(prob_pca, prob_pca_bar)

            # rank loss
            out_label = F.softmax(out_label, dim=1)
            feat_copy = out_label.detach()
            sim_mat = feat_copy.mm(feat_copy.t())
            
            rank_idx_positive = torch.argsort(sim_mat, dim=1, descending=True)
            rank_idx_positive = rank_idx_positive[:, :args.topk]
            rank_idx_negative = torch.argsort(sim_mat, dim=1, descending=False)
            rank_idx_negative = rank_idx_negative[:, :args.topk]
            target_ulb = torch.zeros_like(sim_mat).float().to(args.device)
            for i in range(x.size(0)):
                target_ulb[i, rank_idx_positive[i,:]] = 1
                target_ulb[i, rank_idx_negative[i, :]] = -1
                target_ulb[i, i] = 1
            
            target_ulb = target_ulb.view(-1)
            prob_mat = prob.mm(prob.t())
            prob_mat = prob_mat.view(-1)
            
            rank_loss = bce_criterion(prob_mat, target_ulb)

            # LwF loss
            ref_output = old_model(x)
            soft_target = F.softmax(ref_output / 2, dim=1)
            new_output = old_model.linear(feat_label)
            logp = F.log_softmax(new_output / 2, dim=1)
            reg_loss = -torch.mean(torch.sum(soft_target * logp, dim=1))  # * args.T * args.T

            ## total loss
            loss = sharp_loss + consistency_loss + rank_loss + args.alpha*reg_loss
            
            total_loss_record.update(loss.item(), x.size(0))
            cluster_loss_record.update(sharp_loss.item(), x.size(0))
            rank_loss_record.update(consistency_loss.item(), x.size(0))
            reg_loss_record.update(consistency_loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {}, Center {:.4f}, cluster Loss: {:.4f}, rank Loss: {:.4f}, reg Loss: {:.4f}'.format(epoch,
                torch.mean(new_model.center), cluster_loss_record.avg, rank_loss_record.avg, reg_loss_record.avg))
        
        new_model.eval()
        _, _, _, probs = eval_unlabel_classes(new_model, eva_loader, args)

        if (epoch+1) % args.update_interval == 0: 
            print('updating target ...')
            args.p_targets = target_distribution(probs)

        if epoch%50==0:
            torch.save(new_model.state_dict(), args.model_dir)
            print("model saved to {}.".format(args.model_dir))
            sio.savemat(args.save_clusters_path, {'clusters': new_model.center.data.cpu().numpy()})

def eval_unlabel_classes(model, test_loader, args):
    model.eval()
    preds_pca = np.array([])
    targets = np.array([])
    probs_pca = np.zeros((len(test_loader.dataset), args.n_clusters))
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
            x, label = x.to(args.device), label.to(args.device)
            _, _, out_pca, _ = model(x, flag=2)
            prob_pca = feat2prob(out_pca, model.center)
            _, pred_pca = prob_pca.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds_pca = np.append(preds_pca, pred_pca.cpu().numpy())
            idx = idx.data.cpu().numpy()
            probs_pca[idx, :] = prob_pca.cpu().detach().numpy()
        acc, nmi, ari = cluster_acc(targets.astype(int), preds_pca.astype(int)), nmi_score(targets, preds_pca), ari_score(targets,
                                                                                                                  preds_pca)
        print('[Unlabel Classes] Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
        probs_pca = torch.from_numpy(probs_pca)

    return acc, nmi, ari, probs_pca


def eval_label_classes(model, test_loader, args):
    model.eval()
    preds_pca = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
            x, label = x.to(args.device), label.to(args.device)
            _, out_label, _, _ = model(x, flag=2)
            _, pred_pca = out_label.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds_pca = np.append(preds_pca, pred_pca.cpu().numpy())
        acc, nmi, ari = cluster_acc(targets.astype(int), preds_pca.astype(int)), nmi_score(targets, preds_pca), ari_score(targets,
                                                                                                                  preds_pca)
        print('[Label Classes] Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    return acc, nmi, ari


def eval_all_classes(model, test_loader, args):
    model.eval()
    preds_pca = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
            x, label = x.to(args.device), label.to(args.device)
            _, out_label, out_pca, _ = model(x, flag=2)
            out_pca = feat2prob(out_pca, model.center)
            out_cat = torch.cat([out_label, out_pca], dim=1)
            _, pred_pca = out_cat.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds_pca = np.append(preds_pca, pred_pca.cpu().numpy())
        acc, nmi, ari = cluster_acc(targets.astype(int), preds_pca.astype(int)), nmi_score(targets, preds_pca), ari_score(targets,
                                                                                                                  preds_pca)
        print('[Label Classes] Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    return acc, nmi, ari


def extract_feature(model, test_loader, args):
    model.eval()
    preds_pca = np.array([])
    labels = np.array([])
    targets = np.array([])
    feats = np.zeros((len(test_loader.dataset), 512))
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
            x, label = x.to(args.device), label.to(args.device)
            feat, _ = model(x, flag=1)
            targets = np.append(targets, label.cpu().numpy())
            idx = idx.data.cpu().numpy()
            feats[idx, :] = feat.cpu().detach().numpy()
    sio.savemat(args.save_features_path, {'feat': feats, 'GT': targets})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--warmup_flag', default=False, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--warmup_lr', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', default=30, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=5, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--milestones', default=[40, 60, 80], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--update_interval', default=5, type=int) 
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_txt', default=True, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--pretrain_dir', type=str, default='./results/cifar10_pretrain/cifar10_labeled_classes_pretraining.pth')
    parser.add_argument('--dataset_root', type=str, default='../Datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--model_name', type=str, default='resnet18_cifar10_ResTune.pth')
    parser.add_argument('--save_txt_name', type=str, default='results_resnet18_cifar10_ResTune.txt')
    parser.add_argument('--save_clusters_name', type=str, default='clusters_resnet18_cifar10_ResTune.mat')
    parser.add_argument('--save_feature_name', type=str,
                        default='resnet18_cifar10_ResTune_features.mat')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir + '/' + args.model_name
    args.save_txt_path = args.exp_root + '{}/{}'.format(runner_name, args.save_txt_name)
    args.save_clusters_path = args.exp_root + '{}/{}'.format(runner_name, args.save_clusters_name)
    args.save_features_path = args.exp_root + '{}/{}'.format(runner_name, args.save_feature_name)
    args.alpha = 1

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    unlabeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list = range(args.num_labeled_classes, num_classes))
    unlabeled_val_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
    unlabeled_test_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
    labeled_test_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(args.num_labeled_classes))
    all_test_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(num_classes))

    if args.mode == 'train':
        old_model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes).to(args.device)
        old_model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
        old_model.linear= Identity()
        init_feat_extractor = old_model
        init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, unlabeled_val_loader, args)
        args.p_targets = target_distribution(init_probs)

        new_model = ResNet_unlabel(BasicBlock, [2, 2, 2, 2], nclass_label=args.num_labeled_classes,nclass_unlabel=args.num_unlabeled_classes).to(args.device)
        new_model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
        new_model.center.data = torch.tensor(init_centers).float().to(args.device)
        
        for param in new_model.parameters():
            param.requires_grad = True

        ## Training
        old_model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes).to(args.device)
        old_model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
        for param in old_model.parameters():
            param.requires_grad = False
        train(old_model, new_model, unlabeled_train_loader, unlabeled_val_loader, args)

        ## Save model
        torch.save(new_model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

        ## Test
        acc_unlabel_only, nmi_unlabel_only, ari_unlabel_only, _ = eval_unlabel_classes(new_model, unlabeled_test_loader, args)
        acc_label_only, nmi_label_only, ari_label_only = eval_label_classes(new_model, labeled_test_loader, args)
        acc_unlabel_all, nmi_unlabel_all, ari_unlabel_all = eval_all_classes(new_model, unlabeled_test_loader, args)
        acc_label_all, nmi_label_all, ari_label_all = eval_all_classes(new_model, labeled_test_loader, args)
        acc_test_all, nmi_test_all, ari_test_all = eval_all_classes(new_model, all_test_loader, args)
        print('[Unlabel Test only] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc_unlabel_only, nmi_unlabel_only, ari_unlabel_only))
        print('[Label Test only] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc_label_only, nmi_label_only, ari_label_only))
        print('[Unlabel Test split] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc_unlabel_all, nmi_unlabel_all, ari_unlabel_all))
        print('[Label Test split] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc_label_all, nmi_label_all, ari_label_all))
        print('[ALL Test split] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc_test_all, nmi_test_all, ari_test_all))

        if 1==1:
            with open(args.save_txt_path, 'a') as f:
                #f.write("[Train split] K-means: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}\n".format(init_acc, init_nmi, init_ari))
                f.write("[Unlabel Test only] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}\n".format(acc_unlabel_only, nmi_unlabel_only, ari_unlabel_only))
                f.write("[Label Test only] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}\n".format(acc_label_only, nmi_label_only, ari_label_only))
                f.write("[Unlabel Test split] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}\n".format(acc_unlabel_all, nmi_unlabel_all, ari_unlabel_all))
                f.write("[Label Test split] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}\n".format(acc_label_all, nmi_label_all, ari_label_all))
                f.write("[ALL Test split] ResNovel: ACC {:.4f}, NMI {:.4f}, ARI {:.4f}\n".format(acc_test_all, nmi_test_all, ari_test_all))
    else:
        
        torch.cuda.synchronize()
        time_start = time.time()

        all_test_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(num_classes))
        new_model = ResNet_unlabel(BasicBlock, [2, 2, 2, 2], nclass_label=args.num_labeled_classes,nclass_unlabel=args.num_unlabeled_classes).to(args.device)
        new_model.load_state_dict(torch.load(args.model_dir), strict=False)
        new_model.eval()
        # extract_feature(new_model, unlabeled_test_loader, args)
        acc_test_all, nmi_test_all, ari_test_all = eval_all_classes(new_model, all_test_loader, args)

        torch.cuda.synchronize()
        time_end = time.time()
        print('totally cost',time_end-time_start)
