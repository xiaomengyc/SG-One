

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import shutil
import argparse
import my_optim
from oneshot import *
from utils.LoadDataSeg import data_loader
from utils.Restore import restore
from utils import AverageMeter
from utils.para_number import get_model_para_number

#ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
ROOT_DIR = '/'.join(os.getcwd().split('/'))
print ROOT_DIR


SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
IMG_DIR = os.path.join('/dev/shm/', 'IMAGENET_VOC_3W/imagenet_simple')

LR = 1e-5

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='onemodel_sg-one')
    parser.add_argument("--max_steps", type=int, default=100001)
    parser.add_argument("--lr", type=float, default=LR)
    # parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    # parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)

    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def get_model(args):

    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    print('Number of Parameters: %d'%(get_model_para_number(model)))

    # optimizer
    opti_A = my_optim.get_finetune_optimizer(args, model)

    # if os.path.exists(args.restore_from):

    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    print(args.resume)
    if args.resume:
        restore(snapshot_dir, model)
        print("Resume training...")

    return model, opti_A

def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    return snapshot_dir



def train(args):

    losses = AverageMeter()
    model, optimizer= get_model(args)
    model.train()

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    train_loader = data_loader(args)

    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))

    save_log_dir = get_save_dir(args)
    log_file  = open(os.path.join(save_log_dir, 'log.txt'),'w')


    count = args.start_count
    for dat in train_loader:
        if count > args.max_steps:
            break

        anchor_img, anchor_mask, pos_img, pos_mask, neg_img, neg_mask = dat

        anchor_img, anchor_mask, pos_img, pos_mask, \
            = anchor_img.cuda(), anchor_mask.cuda(), pos_img.cuda(), pos_mask.cuda()

        anchor_mask = torch.unsqueeze(anchor_mask,dim=1)
        pos_mask = torch.unsqueeze(pos_mask, dim=1)

        logits = model(anchor_img, pos_img, neg_img, pos_mask)
        loss_val, cluster_loss, loss_bce = model.get_loss(logits, anchor_mask)

        loss_val_float = loss_val.data.item()

        losses.update(loss_val_float, 1)

        out_str = '%d, %.4f\n'%(count, loss_val_float)
        log_file.write(out_str)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        count += 1
        if count%args.disp_interval == 0:
            # print('Step:%d \t Loss:%.3f '%(count, losses.avg))
            print('Step:%d \t Loss:%.3f \t '
                  'Part1: %.3f \t Part2: %.3f'%(count, losses.avg,
                                        cluster_loss.cpu().data.numpy() if isinstance(cluster_loss, torch.Tensor) else cluster_loss,
                                        loss_bce.cpu().data.numpy() if isinstance(loss_bce, torch.Tensor) else loss_bce))


        if count%args.save_interval == 0 and count >0:
            save_checkpoint(args,
                            {
                                'global_counter': count,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='step_%d.pth.tar'
                                     %(count))
    log_file.close()

if __name__ == '__main__':
    args = get_arguments()
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
