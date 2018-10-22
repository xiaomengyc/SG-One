from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image
import random
# from .transforms import functional
# random.seed(1234)
# from .transforms import functional
import cv2
import math
from .transforms import transforms
from torch.utils.data import DataLoader
from utils.config import cfg
from datasets.factory import get_imdb

class mydataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, args, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_db = get_imdb('voc_2012_train')
        self.img_db.get_seg_items(args.group, args.num_folds)
        self.transform = transform
        self.split = args.split
        self.group = args.group
        self.num_folds = args.num_folds
        self.is_train = is_train


    def __len__(self):
        # return len(self.image_list)
        return 100000000

    def read_img(self, path):
        return cv2.imread(path)

    def _read_data(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat>0] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_val(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_train(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat!=category+1] = 0
        mask_dat[mask_dat==category+1] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def get_item_mlclass_val(self, query_img, sup_img_list):
        que_img, que_mask = self._read_mlclass_val(query_img)
        supp_img = []
        supp_mask = []
        for img_dit in sup_img_list:
            tmp_img, tmp_mask = self._read_mlclass_val(img_dit)
            supp_img.append(tmp_img)
            supp_mask.append(tmp_mask)

        supp_img_processed = []
        if self.transform is not None:
            que_img = self.transform(que_img)
            for img in supp_img:
                supp_img_processed.append(self.transform(img))

        return que_img, que_mask, supp_img_processed, supp_mask

    def get_item_mlclass_train(self, query_img, support_img, category):
        que_img, que_mask = self._read_mlclass_train(query_img, category)
        supp_img, supp_mask = self._read_mlclass_train(support_img, category)
        if self.transform is not None:
            que_img = self.transform(que_img)
            supp_img = self.transform(supp_img)

        return que_img, que_mask, supp_img, supp_mask

    def get_item_single_train(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        thrid_img, thrid_mask = self._read_data(dat_dicts[2])

        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            thrid_img = self.transform(thrid_img)

        return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask

    def get_item_rand_val(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        thrid_img, thrid_mask = self._read_data(dat_dicts[2])

        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            thrid_img = self.transform(thrid_img)

        # return first_img, first_mask, second_img,second_mask
        return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask, dat_dicts

    def __getitem__(self, idx):
        if self.split == 'train':
            dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)
            return self.get_item_single_train(dat_dicts)
        elif self.split == 'random_val':
            dat_dicts = self.img_db.get_triple_images(split='val', group=self.group, num_folds=4)
            return self.get_item_rand_val(dat_dicts)
        elif self.split == 'mlclass_val':
            query_img, sup_img_list = self.img_db.get_multiclass_val(split='val', group=self.group, num_folds=4)
            return self.get_item_mlclass_val(query_img, sup_img_list)
        elif self.split == 'mlclass_train':
            query_img, support_img, category = self.img_db.get_multiclass_train(split='train', group=self.group, num_folds=4)
            return self.get_item_mlclass_train(query_img, support_img, category)


    # def __getitem__(self, idx):
    #     if self.split == 'train':
    #         dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)
    #
    #     first_img, first_mask = self._read_data(dat_dicts[0])
    #     second_img, second_mask = self._read_data(dat_dicts[1])
    #     thrid_img, thrid_mask = self._read_data(dat_dicts[2])
    #
    #     if self.transform is not None:
    #         first_img = self.transform(first_img)
    #         second_img = self.transform(second_img)
    #         thrid_img = self.transform(thrid_img)
    #
    #     return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask


