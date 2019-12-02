import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
nnquery_module = tf.load_op_library(os.path.join(base_dir, 'tf_dist2weight_so.so'))

# def dist2weight(nn_count, nn_dist, radius, type='inv_dist'):
#     '''
#     The dist2weight function returns normalized inverse distance as weights if type
#     is inv_dist. otherwise, it returns the bilinear_like weights. See the .cu file
#     for details of the weight generation.
#
#     Input:
#         nn_count: (batch, mpoint) int32 array, number of neighbors
#         nn_dist: (batch, mpoint, nnsample) float32, sqrt distance array
#         radius: float32, it is only required for bilinear_like computation
#         type: string, should be one of {'inv_dist','bilinear_like'}
#     Output:
#         weight: (batch, mpoint, nnsample) float32, weights for interpolation
#     '''
#     if type=='inv_dist':
#         weight = nnquery_module.build_sample_weight(nn_count, nn_dist, radius, 0)
#     else:
#         weight = nnquery_module.build_sample_weight(nn_count, nn_dist, radius, 1)
#
#     epsilon = 1e-15
#     sum_weight = tf.reduce_sum(weight, axis=-1, keepdims=True)
#     normalized_weight = tf.divide(weight, sum_weight+epsilon)
#     return normalized_weight
# ops.NoGradient('BuildSampleWeight')

def dist2weight(nn_count,nn_dist):
    epsilon = 1e-15
    inv_dist = tf.math.reciprocal(nn_dist+epsilon)
    sum_inv_dist = tf.reduce_sum(inv_dist, axis=-1, keepdims=True)
    weight = tf.divide(inv_dist, sum_inv_dist+epsilon)

    return weight


