# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
import pdb

np.random.seed(1234)

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._image_index = []
        # self.group=0
        # self.num_folds= 4
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')   # method=self.gt_roidb
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()  # call self.gt_roidb()
        return self._roidb

    # @property
    # def cache_path(self):
    #     cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    #     if not os.path.exists(cache_path):
    #         os.makedirs(cache_path)
    #     return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def image_id_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def get_pair_images(self):
        self.group=0
        self.num_folds= 4
        cats = self.get_cats(self.split, self.fold)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        sample_img_ids = np.random.choice(len(self.grouped_imgs[rand_cat]), 2, replace=False)
        return (self.grouped_imgs[rand_cat][sample_img_ids[0]],
                self.grouped_imgs[rand_cat][sample_img_ids[1]])

    def get_triple_images(self, split, group, num_folds=4):
        cats = self.get_cats(split, group, num_folds)
        rand_cat = np.random.choice(cats, 2, replace=False)
        sample_img_ids_1 = np.random.choice(len(self.grouped_imgs[rand_cat[0]]), 2, replace=False)
        sample_img_ids_2 = np.random.choice(len(self.grouped_imgs[rand_cat[1]]), 1, replace=False)

        anchor_img = self.grouped_imgs[rand_cat[0]][sample_img_ids_1[0]]
        pos_img = self.grouped_imgs[rand_cat[0]][sample_img_ids_1[1]]
        neg_img = self.grouped_imgs[rand_cat[1]][sample_img_ids_2[0]]

        return (anchor_img, pos_img, neg_img)

    def get_multiclass_train(self, split, group, num_folds=4):
        cats = self.get_cats('train', group, num_folds)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        cat_list = self.multiclass_grouped_imgs[rand_cat]
        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)
        query_img = cat_list[sample_img_ids_1[0]]
        support_img = cat_list[sample_img_ids_1[1]]
        return query_img, support_img, rand_cat

    def get_multiclass_val(self, split, group, num_folds=4):
        cats = self.get_cats('val', group, num_folds)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        cat_list = self.multiclass_grouped_imgs[rand_cat]
        sample_img_ids_1 = np.random.choice(len(cat_list), 1, replace=False)[0]
        query_img = cat_list[sample_img_ids_1]
        sup_img_list=[]
        for cat_id in cats:
            cat_list = self.grouped_imgs[cat_id]
            sample_img_ids_1 = np.random.choice(len(cat_list), 1, replace=False)[0]
            img_dict = cat_list[sample_img_ids_1]
            sup_img_list.append(img_dict)
        return (query_img, sup_img_list)


    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

