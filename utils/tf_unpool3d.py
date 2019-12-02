import tensorflow as tf


def mean_interpolate(input, nn_index):
    batch_size = tf.shape(nn_index)[0]
    num_pt = tf.shape(nn_index)[1]
    num_nn = tf.shape(nn_index)[2]
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_pt, num_nn, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(nn_index,axis=3)], axis=3)
    output = tf.gather_nd(input, indices)
    output = tf.reduce_mean(output, axis=2)
    return output


def weighted_interpolate(input, weight, nn_index):
    batch_size = tf.shape(nn_index)[0]
    num_pt = tf.shape(nn_index)[1]
    num_nn = tf.shape(nn_index)[2]
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_pt, num_nn, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(nn_index, axis=3)], axis=3)
    output = tf.gather_nd(input, indices)
    output = tf.transpose(output, perm=[0, 1, 3, 2])
    weight = tf.expand_dims(weight, axis=3)
    output = tf.matmul(output, weight)
    output = tf.squeeze(output)
    return output

