#!/usr/bin/env bash

# the results of  compute self-convolution coefficient as sc = max(0.0,1 - dist/radius)

export CUDA_VISIBLE_DEVICES=0
python evaluate_s3dis_with_overlap.py --model=SPH3D_s3dis_resnet_fuzzy --config=s3dis_config_resnet --log_dir=SPH3D_fuzzy_radius --model_name=model.ckpt-109