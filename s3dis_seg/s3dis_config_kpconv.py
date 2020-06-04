import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/buildkernel'))

from kpconv_points import kernel_point_optimization_debug
import numpy as np

num_input = 8192
num_cls = 13

mlp = 64
num_sample = [2048, 768, 384, 128]
print(num_sample)
radius = [0.1, 0.2, 0.4, 0.8]
nn_uplimit = [64, 64, 64, 64]
channels = [128, 256, 512, 1024]

weight_decay = None

binSize = 15
kernel_points, _ = kernel_point_optimization_debug(1.0, binSize)

normalize = True
pool_method = 'max'
unpool_method = 'weighted'
nnsearch = 'sphere'
sample = 'FPS' #{'FPS','IDS','random'}

with_bn = True
with_bias = False