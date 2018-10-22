#!/usr/bin/env bash

cd ..

GPU_ID=2
GPOUP_ID=2

CUDA_VISIBLE_DEVICES=$GPU_ID python train_frame.py \
    --group=${GPOUP_ID} \
    --num_folds=4 \
    --arch=onemodel_sg-one \
    --lr=1e-5
