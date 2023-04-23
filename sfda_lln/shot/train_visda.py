import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import shot_loss
from torch.utils.data import DataLoader
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, f1_score
import common.vision.datasets as datasets
from vis_sourcefree import *


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
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
        transforms.ToTensor(),
        normalize
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
        transforms.ToTensor(),
        normalize
    ])


class ELR_loss(nn.Module):
    def __init__(self, beta, lamb, num, cls):
        super(ELR_loss, self).__init__()
        # self.pseudo_targets = torch.empty(args.nb_samples, dtype=torch.long).random_(args.nb_classes).cuda()
        self.ema = torch.zeros(num, cls).cuda()
        self.beta = beta
        self.lamb = lamb

    def forward(self, index, outputs):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * (
                    (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg
        return final_loss


def data_load(args):
    ## prepare data
    dset_loaders = {}
    train_bs = args.batch_size

    args.source = 'Synthetic'
    args.target = 'Real'

    roots = os.path.join(args.root, 'validation')

    dataset = datasets.__dict__['VisDA2017']
    train_source_dataset = dataset(root=roots, task=args.target, r=0, download=False, transform=image_train())

    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes

    val_dataset = dataset(root=roots, task=args.target, r=0, download=False, transform=image_test())

    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)

    train_source_loader = DataLoader(train_source_dataset, batch_size=train_bs,
                                     shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_bs, shuffle=False, num_workers=args.num_workers)

    dset_loaders["target"] = train_source_loader
    dset_loaders["test"] = val_loader

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(shot_loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        #matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = f1_score(all_label, torch.squeeze(predict).float(), average=None)
        # acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    criterion_elr = ELR_loss(args.beta, args.lamb, args.nb_samples, args.nb_classes)

    # netF = ResNet_FE().cuda()
    # netC = feat_classifier(class_num=args.class_num).cuda()

    netF = ResBase().cuda()

    netB = feat_bootleneck(feature_dim=netF.in_features).cuda()

    netC = feat_classifier(class_num=args.class_num).cuda()

    weights_dir = args.source.lower()[0] + '2' + args.target.lower()[0]
    args.name = weights_dir
    netF.load_state_dict(torch.load('~/da_weights/visda/source/uda/visda-2017/T/source_F.pt'))
    netB.load_state_dict(torch.load('~/da_weights/visda/source/uda/visda-2017/T/source_B.pt'))
    netC.load_state_dict(torch.load('~/da_weights/visda/source/uda/visda-2017/T/source_C.pt'))

    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1.0}]


    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)



    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best = 0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            classifier_loss = criterion_elr(tar_idx, outputs_test) + classifier_loss
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(shot_loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = 2*torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss



        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            best = max(best, acc_s_te)
            print(log_str + '\n')
            netF.train()
            netB.train()

    print("best acc is", best)
    with open('vis_elr_f1.txt', 'a') as f:
        f.write('beta{}\tlamb{}\tcls_par0.3\tsrc/tar:[{}/{}]\t{}\n'.format(args.beta, args.lamb, args.source, args.target, best))

    return netF, netB, netC, best


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    print(log_str + '\n')
    # pred_label = predict.numpy().astype('int')
    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('-s', '--source', default='Pr', help='source domain(s)')
    parser.add_argument('-t', '--target', default='Ar', help='target domain(s)')

    # model dataset
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    # source model
    # parser.add_argument('srcpth', type=str, help='path for source models')

    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='res101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--beta', default=0.6, type=float, help='ema for t')
    parser.add_argument('--lamb', default=25.0, type=float, help='elr loss hyper parameter')

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)

    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    print(args)
    train_target(args)
