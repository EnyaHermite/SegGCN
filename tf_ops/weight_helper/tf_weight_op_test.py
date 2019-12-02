import tensorflow as tf
import sys, os, math
import numpy as np
from termcolor import colored
from weight_helper import dist2weight

M_PI = np.float32(3.14159265358979323846)  # pi
M_EPS = np.float32(1.19209e-4)             # epsilon

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir,'nnquery'))
from tf_nnquery import build_sphere_neighbor, build_nearest_neighbor


def check_weight_inv(nnCount,nnDist):
    print(nnCount.shape,nnDist.shape)
    B,N,K = nnDist.shape

    weight = np.zeros(nnDist.shape)
    for b in range(B):
        for n in range(N):
            sumW = 0
            K = nnCount[b,n]
            for k in range(K):
                weight[b,n,k] = 1/max(nnDist[b,n,k],1e-15)
                sumW += weight[b,n,k]

            for k in range(K):
                weight[b,n,k] /= (sumW+1e-15)

    return weight


def check_weight_bilinear(nnCount,nnDist,radius):
    print(nnCount.shape,nnDist.shape)
    B,N,K = nnDist.shape

    weight = np.zeros(nnDist.shape)
    for b in range(B):
        for n in range(N):
            sumW = 0
            K = nnCount[b,n]
            for k in range(K):
                weight[b,n,k] = max(0,1-nnDist[b,n,k]/radius)
                sumW += weight[b,n,k]

            for k in range(K):
                weight[b,n,k] /= (sumW+1e-15)

    return weight


if __name__=='__main__':
    import time

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    # np.random.seed(100)
    B, N, M = 64, 256, 512
    radius, nnsample = 0.2, 64
    kernel = [8, 2, 3]
    tmp1 = np.random.random((B, N, 3)).astype('float32')
    tmp2 = np.random.random((B, M, 3)).astype('float32')
    for i in range(B):
        xyz = tmp1[i,...]
        xyz = xyz - np.mean(xyz, axis=0)
        scale = np.sqrt(np.amax(np.sum(np.square(xyz), axis=1)))
        xyz /= scale
        tmp1[i,...] = xyz

        xyz = tmp2[i, ...]
        xyz = xyz - np.mean(xyz, axis=0)
        scale = np.sqrt(np.amax(np.sum(np.square(xyz), axis=1)))
        xyz /= scale
        tmp2[i, ...] = xyz

    maxIter = 100
    with tf.device('/gpu:0'):
        database = tf.constant(tmp1)
        query = tf.constant(tmp2)
        nn_index, nn_count, nn_dist = build_nearest_neighbor(database, query)
        weight = dist2weight(nn_count, nn_dist)
    with tf.Session(config=config) as sess:
        now = time.time()
        for _ in range(maxIter):
            nnIndex, nnCount, nnDist, Weight = sess.run([nn_index, nn_count, nn_dist, weight])
        cpp_gpu_time = (time.time() - now) / maxIter
    print(cpp_gpu_time)

    now = time.time()
    check_weight = check_weight_inv(nnCount,nnDist)
    cpu_time = time.time() - now
    print(cpp_gpu_time, cpu_time)

    T = np.sum(check_weight,axis=-1)
    print(T.shape,T[0:5,0:10],np.amin(T),np.amax(T))

    print(nnDist.shape, Weight.shape, check_weight.shape)
    print(np.isnan(check_weight).any(),np.isnan(Weight).any())
    if np.allclose(check_weight,Weight,atol=1e-4):
        print(colored('dist2weight passed the test!','green'))
    else:
        print(colored('dist2weight DID NOT pass the test!', 'red'))

    print("===================================The End====================================")

    # for type in ['inv_dist','bilinear_like']:
    #     print("========================testing the dist2weight function of type %s======================="%type)
    #     maxIter = 100
    #     with tf.device('/gpu:0'):
    #         database = tf.constant(tmp1)
    #         query = tf.constant(tmp2)
    #         nn_index, nn_count, nn_dist = build_sphere_neighbor(database, query, radius=radius, nnsample=nnsample)
    #         weight = dist2weight(nn_count, nn_dist, radius, type)
    #     with tf.Session(config=config) as sess:
    #         now = time.time()
    #         for _ in range(maxIter):
    #             nnIndex, nnCount, nnDist, Weight = sess.run([nn_index, nn_count, nn_dist, weight])
    #         cpp_gpu_time = (time.time() - now) / maxIter
    #     print(cpp_gpu_time)
    #
    #     if type=='inv_dist':
    #         now = time.time()
    #         check_weight = check_weight_inv(nnCount,nnDist)
    #         cpu_time = time.time() - now
    #     else:
    #         now = time.time()
    #         check_weight = check_weight_bilinear(nnCount, nnDist,radius)
    #         cpu_time = time.time()-now
    #     print(cpp_gpu_time, cpu_time)
    #
    #     T = np.sum(check_weight,axis=-1)
    #     print(T.shape,T[0:5,0:10],np.amin(T),np.amax(T))
    #
    #     print(nnDist.shape, Weight.shape, check_weight.shape)
    #     print(np.isnan(check_weight).any(),np.isnan(Weight).any())
    #     if np.allclose(check_weight,Weight,atol=1e-4):
    #         print(colored('dist2weight type %s passed the test!'%type,'green'))
    #     else:
    #         print(colored('dist2weight type %s DID NOT pass the test!'%type, 'red'))
    #
    #     print("===================================The End====================================")