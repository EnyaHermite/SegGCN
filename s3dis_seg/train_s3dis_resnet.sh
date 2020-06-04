#!/usr/bin/env bash

# the results of  compute self-convolution coefficient as sc = max(0.0,1 - dist/radius)

export CUDA_VISIBLE_DEVICES=1
python train_s3dis_old.py  --log_dir=mean

