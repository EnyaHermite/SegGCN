import tensorflow as tf
import sys, os
import numpy as np
from termcolor import colored

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'tf_ops/nnquery'))

print(root_dir)

from tf_unpool3d import mean_interpolate
from tf_nnquery import build_nearest_neighbor

def check_mean_interpolate(input, nn_index, nn_count):
    B, M, C = input.shape
    if len(nn_index.shape)==4:
        nn_index = nn_index[:,:,:,0]
    _, N, _ = nn_index.shape

    output = np.zeros((B, N, C), dtype=np.float32)

    for b in range(B):
        for n in range(N):
            K = nn_count[b,n]
            for c in range(C):
                for k in range(K):
                    m = nn_index[b,n,k]
                    output[b, n, c] += input[b, m, c]/K

    return output

def check_mean_interpolate_grad(input, grad_output, nn_index, nn_count):
    B, N, C = grad_output.shape
    _, M, _ = input.shape

    if len(nn_index.shape)==4:
        nn_index = nn_index[:,:,:,0]

    grad_input = np.zeros((B, M, C), dtype=np.float32)

    for b in range(B):
        for n in range(N):
            K = nn_count[b, n]
            for c in range(C):
                for k in range(K):
                    m = nn_index[b,n,k]
                    grad_input[b,m,c] += grad_output[b,n,c]/K
    return grad_input


if __name__=='__main__':
    import time

    # np.random.seed(100)
    B, N, M, C, radius, nnsample = (32, 128, 48, 64, 0.2, 32)
    kernel = [8, 2, 3]
    binSize = np.prod(kernel) + 1

    tmp0 = np.random.random((B, N, 3)).astype('float32')
    tmp1 = tmp0[:,np.random.choice(N,M,False),:]
    tmp2 = np.random.random((B, N, C)).astype('float32')
    tmp3 = np.random.random((B, M, C)).astype('float32')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    print(tmp0.shape, tmp1.shape)
    with tf.device('/gpu:0'):
        xyz0 = tf.constant(tmp0)
        xyz1 = tf.constant(tmp1)
        nn_index, nn_count, nn_dist = build_nearest_neighbor(xyz1, xyz0)
    with tf.Session(config=config) as sess:
        unpoolIndex, unpoolCount = sess.run([nn_index, nn_count])

    print(unpoolIndex.shape, unpoolCount.shape)

    with tf.device('/gpu:0'):
        featIn = tf.constant(tmp3)
        gradOut = tf.constant(tmp2)
        feat_out = mean_interpolate(featIn, nn_index)#unpoolIndex)
        grad_in = tf.gradients(feat_out, featIn, gradOut)

    with tf.Session(config=config) as sess:
        now = time.time()
        for _ in range(100):
            gradIn, featOut = sess.run([grad_in, feat_out])
        cpp_gpu_time = (time.time() - now) / 100

    now = time.time()
    feat_out_cpu = check_mean_interpolate(tmp3, unpoolIndex, unpoolCount)
    grad_input_cpu = check_mean_interpolate_grad(tmp3, tmp2, unpoolIndex, unpoolCount)
    print(time.time() - now, cpp_gpu_time)

    if np.allclose(feat_out_cpu, featOut) and np.allclose(grad_input_cpu, gradIn):
        print(colored("mean-interpolation test passed!",'green'))






