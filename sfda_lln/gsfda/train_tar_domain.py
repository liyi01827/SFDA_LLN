import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list_domain import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


class ELR_loss(nn.Module):
    def __init__(self, beta, lamb, num, cls):
        super(ELR_loss, self).__init__()
        self.ema = torch.zeros(num, cls).cuda()
        self.beta = beta
        self.lamb = lamb



    def forward(self, index,  outputs):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg
        return final_loss


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size


    txt_test = open(args.test_dset_path).readlines()


    txt_tar = open(args.t_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_test, transform=image_train(), tmp=args.root)
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)

    dsets["test"] = ImageList_idx(txt_tar, transform=image_test(), tmp=args.root)
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC,t=0, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs),t=t)[0])
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent



def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck_sdaE(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = '~/da_weights/gsfda/domainnet/{}/source_F.pt'.format(args.source)
    netF.load_state_dict(torch.load(modelpath))
    modelpath = '~/da_weights/gsfda/domainnet/{}/source_B.pt'.format(args.source)
    netB.load_state_dict(torch.load(modelpath))
    modelpath = '~/da_weights/gsfda/domainnet/{}/source_C.pt'.format(args.source)
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        if k.find('bn')!=-1:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        #if k.find('em')==-1: # the embedding layer can be either trained or not
        if True:
            param_group += [{'params': v, 'lr': args.lr * 1}] 
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    best=0
    #building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    criterion_elr = ELR_loss(args.beta, args.lamb, num_sample, args.class_num)

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output, _ = netB(netF(inputs), t=1)  # a^t
            output_norm=F.normalize(output)
            outputs = netC(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()


    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0


    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test, masks = netB(netF(inputs_test),t=1)
        masks_old = masks
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm=F.normalize(features_test)
            fea_bank[tar_idx].fill_(-0.1)    #do not use the current mini-batch in fea_bank
            output_f_=output_f_norm.cpu().detach().clone()
            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=5)
            score_near = score_bank[idx_near]    #batch x K x num_class
            score_near=score_near.permute(0,2,1)

            # update banks
            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()  #.cpu()

        const=torch.log(torch.bmm(output_re,score_near)).sum(-1)
        loss=-torch.mean(const)

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + args.epsilon))
        loss += gentropy_loss

        loss += criterion_elr(tar_idx, outputs_test)

        optimizer.zero_grad()
        loss.backward()

        for n, p in netB.bottleneck.named_parameters():
            if n.find('bias') == -1:
                mask_ = ((1 - masks_old)).view(-1, 1).expand(256, 2048).cuda()
                p.grad.data *= mask_
            else:  #no bias here
                mask_ = ((1 - masks_old)).squeeze().cuda()
                p.grad.data *= mask_

        for n, p in netC.named_parameters():
            if n.find('weight_v') != -1:
                masks__=masks_old.view(1,-1).expand(40,256)
                mask_ = ((1 - masks__)).cuda()
                p.grad.data *= mask_

        for n, p in netB.bn.named_parameters():
            mask_ = ((1 - masks_old)).view(-1).cuda()
            p.grad.data *= mask_

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'domainnet':
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB,
                                             netC,t=1,flag= False)

                log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                    args.name, iter_num, max_iter, acc_s_te
                ) + '\n'

            best = max(best, acc_s_te)
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

    with open('./domain_accs/domain_elr_detail.txt', 'a') as f:
        f.write('beta{}\tlamb{}\tname{}\t{}\n'.format(args.beta, args.lamb, args.name, best))

    return best


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ours')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='8',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=40,
                        help="max iterations")
    parser.add_argument('--interval', type=int, default=40)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=8,
                        help="number of workers")
    parser.add_argument(
        '--dset',
        type=str,
        default='domainnet')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net',
                        type=str,
                        default='resnet50')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--beta', default=0.8, type=float, help='ema for t')
    parser.add_argument('--lamb', default=12.0, type=float, help='elr loss hyper parameter')
    parser.add_argument('--all', type=bool, default=False)
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='visda/target/')
    parser.add_argument('--output_src', type=str, default='visda/source/')
    parser.add_argument('--da',
                        type=str,
                        default='uda')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'domainnet':
        names = ['c', 'p', 'r', 's']
        args.class_num = 40

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    image_list = {
        "c": "clipart",
        "i": "infograph",
        "p": "painting",
        "q": "quickdraw",
        "r": "real",
        "s": "sketch",
    }

    if args.all:
        task = ['c', 'p', 'r', 's']
        acc_acg = []
        for my_source in task:
            task1 = {'c', 'p', 'r', 's'}
            task1_rm = {my_source}
            task2 = list(task1.difference(task1_rm))
            for my_target in task2:
                args.source = my_source
                args.target = my_target

                args.t_dset_path = '~/data/domainnet/{}_train.txt'.format(image_list[args.target])
                args.test_dset_path = '~/data/domainnet/{}_test.txt'.format(image_list[args.target])

                args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
                args.name = args.source + str(2) + args.target
                print(args)
                accs = train_target(args)
                acc_acg.append(accs)

        meanacc = sum(acc_acg) / len(acc_acg)
        with open('./domain_accs/domain_elr.txt', 'a') as f:
            f.write('lr{}\tbeta{}\tlamb{}\t{}\n'.format(args.lr, args.beta, args.lamb,  meanacc))

    else:
        args.source = names[args.s]
        args.target = names[args.t]

        args.t_dset_path = '~/data/domainnet/{}_train.txt'.format(image_list[args.target])
        args.test_dset_path = '~/data/domainnet/{}_test.txt'.format(image_list[args.target])
        args.output_dir_src = osp.join(args.output_src, args.da, args.dset,
                                       names[args.s][0].upper())
        args.name = args.source + str(2) + args.target
        train_target(args)

        accs = train_target(args)

