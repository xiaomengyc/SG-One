

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
from utils import Metrics
from tqdm import tqdm
from utils.save_mask import mask_to_img

#ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
ROOT_DIR = '/'.join(os.getcwd().split('/'))
print ROOT_DIR


SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots_mlcls')
# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
IMG_DIR = os.path.join('/dev/shm/', 'IMAGENET_VOC_3W/imagenet_simple')

LR = 1e-5

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='onemodel_v25')
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)

    parser.add_argument("--split", type=str, default='mlclass_val')
    parser.add_argument('--num_folds', type=int, default=4)

    parser.add_argument('--restore_step', type=int, default=10000)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def restore(args, model, group):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(group, args.num_folds))
    filename='step_%d.pth.tar'%(args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s'%(snapshot))

def get_model(args):

    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    print('Number of Parameters: %d'%(get_model_para_number(model)))

    return model

def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    return snapshot_dir

def get_org_img(img):
    img = np.transpose(img, (1,2,0))
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img = img*std_vals + mean_vals
    img = img*255
    return img

def val(args):

    model= get_model(args)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    # if not os.path.exists(get_save_dir(args)):
    #     os.makedirs(get_save_dir(args))

    hist = np.zeros((21, 21))

    for group in range(4):
        args.group = group
        print("="*20 + "GROUP %d"%(args.group)+"="*20)
        restore(args, model, args.group)
        pbar = tqdm(total=args.max_steps)
        pbar.set_description('GROUP %d'%(args.group))
        train_loader = data_loader(args)

        count = 0
        for dat in train_loader:
            count += 1
            pbar.update(1)
            if count > args.max_steps:
                break

            que_img, que_mask, supp_img, supp_mask = dat

            que_img = que_img.cuda()
            # org_img = get_org_img(que_img.squeeze().cpu().data.numpy())
            # cv2.imwrite('query.png', org_img)

            cat_values = 0
            pred_sum = 0
            for i in range(5):
                pos_img = supp_img[i].cuda()
                pos_mask = supp_mask[i].cuda()
                pos_mask[pos_mask>0.] = 1.
                pos_mask = torch.unsqueeze(pos_mask, dim=1)


                logits = model(que_img, pos_img, None, pos_mask)
                out_softmax, pred = model.get_pred(logits, que_img)

                pred_sum += pred
                if i == 0:
                    cat_values = out_softmax
                    cat_values[0,:,:] = cat_values[0,:,:]*0.
                else:
                    cat_values = torch.cat((cat_values, out_softmax[1,:,:].unsqueeze(dim=0)), dim=0)


            val, pred = torch.max(cat_values, dim=0)
            pred_sum[pred_sum>0.] = 1.0
            pred = pred + args.group*5
            pred = pred_sum*pred
            tmp_pred = pred.cpu().data.numpy()
            hist += Metrics.fast_hist(tmp_pred.astype(np.int32), que_mask.squeeze().data.numpy().astype(np.int32), 21)


            org_img = get_org_img(que_img.squeeze().cpu().data.numpy())
            img = mask_to_img(tmp_pred, org_img)
            cv2.imwrite('save_bins/que_pred/query_%d.png'%(count), img)
                # org_img = get_org_img(pos_img.squeeze().cpu().data.numpy())
                # cv2.imwrite('supp_%d.png'%(i), org_img)
                #
                # np_pred = pred.cpu().data.numpy()
                # cv2.imwrite('%d.png'%(i), np_pred*255)
        miou = Metrics.get_voc_iou(hist)
        print('IOU:', miou)
        print("BMVC:",np.mean(miou[group*5+1:(group+1)*5+1]))
        pbar.close()



    print("="*20 + "Overall"+"="*20)
    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou), np.mean(miou[1:]))

    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(),hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)

if __name__ == '__main__':
    args = get_arguments()
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
