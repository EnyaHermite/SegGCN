import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import sph3gcn_util_fuzzy as s3g_util


def normalize_xyz(points):
    min_xyz = tf.reduce_min(points, axis=1, keepdims=True)
    max_xyz = tf.reduce_max(points, axis=1, keepdims=True)
    center = (max_xyz+min_xyz)/2
    xy = points[:,:,0:2] - center[:,:,0:2]
    z = points[:, :, 2:] # keep the range of height dimension as [0, xxx]
    points = tf.concat((xy, z), axis=2)

    return points


def _IdentityBlock(input, num_out_channel, bin_size, nn_index, nn_count, filt_index,
                   filt_coeff, name, weight_decay=None, reuse=None, with_bn=True,
                   with_bias=True, is_training=None):
    net = s3g_util.pointwise_conv3d(input, num_out_channel//2, '%s_a'%name,
                                    weight_decay=weight_decay,
                                    with_bn=with_bn, with_bias=with_bias,
                                    reuse=reuse, is_training=is_training)
    net = s3g_util.separable_conv3d(net, num_out_channel//2, bin_size, 1, '%s_b'%name, nn_index, nn_count,
                                    filt_index, filt_coeff, weight_decay=weight_decay, with_bn=with_bn,
                                    with_bias=with_bias, reuse=reuse, is_training=is_training)
    net = s3g_util.pointwise_conv3d(net, num_out_channel*2, '%s_c'%name,
                                    weight_decay=weight_decay, with_bn=with_bn,
                                    with_bias=with_bias, reuse=reuse, is_training=is_training)
    return (net+input)


def _ConvBlock(input, num_out_channel, bin_size, nn_index, nn_count, filt_index,
               filt_coeff, name, weight_decay=None, reuse=None, with_bn=True,
               with_bias=True, is_training=None):
    net = s3g_util.pointwise_conv3d(input, num_out_channel//2, name+'2a',
                                    weight_decay=weight_decay,
                                    with_bn=with_bn, with_bias=with_bias,
                                    reuse=reuse, is_training=is_training)
    net = s3g_util.separable_conv3d(net, num_out_channel//2, bin_size, 1, name+'2b', nn_index, nn_count,
                                    filt_index, filt_coeff, weight_decay=weight_decay, with_bn=with_bn,
                                    with_bias=with_bias, reuse=reuse, is_training=is_training)
    net = s3g_util.pointwise_conv3d(net, num_out_channel*2, name+'2c',
                                    weight_decay=weight_decay, with_bn=with_bn,
                                    with_bias=with_bias, reuse=reuse, is_training=is_training)

    shortcut = s3g_util.pointwise_conv3d(input, num_out_channel*2, name+'1',
                                         weight_decay=weight_decay, with_bn=with_bn,
                                         with_bias=with_bias, reuse=reuse, is_training=is_training)
    return (net+shortcut)


def _resnet_block(net, num_out_channels, num_blocks, bin_size, nn_index, nn_count,
                  filt_index, filt_coeff, name, weight_decay=None, reuse=None,
                  with_bn=True, with_bias=True, is_training=None):
    scope = 'resnet_'+name+'_1'
    print(scope)
    net = _ConvBlock(net, num_out_channels, bin_size, nn_index, nn_count,
                     filt_index, filt_coeff, scope, weight_decay=weight_decay, with_bn=with_bn,
                     with_bias=with_bias, reuse=reuse, is_training=is_training)
    for i in range(1,num_blocks):
        scope = 'resnet_'+name+'_%d'%(i+1)
        print(scope)
        net = _IdentityBlock(net, num_out_channels, bin_size, nn_index, nn_count, filt_index,
                             filt_coeff, scope, weight_decay=weight_decay, with_bn=with_bn,
                             with_bias=with_bias, reuse=reuse, is_training=is_training)
    return net


def get_model(points, is_training, config=None):
    end_points = {}
    xyz = points[:, :, 0:3]
    if config.normalize:
        norm_xyz = normalize_xyz(xyz)

    reuse = None
    net = tf.concat((norm_xyz,points[:,:,6:]),axis=2)
    print('input',net)
    net = s3g_util.pointwise_conv3d(net, config.mlp, 'mlp1',
                                    weight_decay=config.weight_decay,
                                    with_bn=config.with_bn, with_bias=config.with_bias,
                                    reuse=reuse, is_training=is_training)

    xyz_layers = []
    encoder = []
    xyz_layers.append(xyz)
    # ===============================================Encoder================================================
    for l in range(len(config.radius)):
        intra_idx, intra_cnt, \
        intra_dst, indices = s3g_util.build_graph(xyz, config.radius[l], config.nn_uplimit[l],
                                                  config.num_sample[l], sample_method=config.sample)
        filt_index, filt_coeff = s3g_util.fuzzy_spherical_kernel(xyz, xyz, intra_idx, intra_cnt,
                                                                 intra_dst, config.radius[l], kernel=config.kernel)
        print(filt_index, filt_coeff)
        net = _resnet_block(net, config.channels[l], config.num_blocks[l], config.binSize, intra_idx,
                            intra_cnt, filt_index, filt_coeff, 'conv'+str(l+1),
                            weight_decay=config.weight_decay, with_bn=config.with_bn,
                            with_bias=config.with_bias, is_training=is_training, reuse=reuse)

        encoder.append(net)
        if config.num_sample[l]>1:
            # ==================================gather_nd====================================
            xyz = tf.gather_nd(xyz, indices)
            xyz_layers.append(xyz)
            inter_idx = tf.gather_nd(intra_idx, indices)
            inter_cnt = tf.gather_nd(intra_cnt, indices)
            inter_dst = tf.gather_nd(intra_dst, indices)
            # =====================================END=======================================

            net = s3g_util.pool3d(net, inter_idx, inter_cnt,
                                  method=config.pool_method, scope='pool'+str(l+1))
    # ===============================================The End================================================

    config.radius.reverse()
    config.nn_uplimit.reverse()
    config.channels.reverse()
    xyz_layers.reverse()
    encoder.reverse()
    print(encoder)
    # ===============================================Decoder================================================
    for l in range(len(config.radius)):
        xyz = xyz_layers[l]
        xyz_unpool = xyz_layers[l+1]

        intra_idx, intra_cnt, intra_dst, \
        inter_idx, inter_cnt, inter_dst = s3g_util.build_graph_deconv(xyz, xyz_unpool,
                                                                      config.radius[l],
                                                                      config.nn_uplimit[l])
        net = s3g_util.pointwise_conv3d(net, config.channels[l], 'deconvMLP'+str(l+1),
                                        weight_decay=config.weight_decay,
                                        with_bn=config.with_bn, with_bias=config.with_bias,
                                        reuse=reuse, is_training=is_training)

        net = s3g_util.unpool3d(net, inter_idx, inter_cnt, inter_dst,
                                method=config.unpool_method, scope='unpool'+str(l+1))
        net = tf.concat((net,encoder[l]),axis=2)
    # ===============================================The End================================================
    end_points['feats'] = net

    # point-wise classifier
    net = s3g_util.pointwise_conv3d(net, config.num_cls, scope='logits', with_bn=False,
                                    with_bias=config.with_bias,
                                    activation_fn=None, is_training=is_training)

    return net, end_points


def get_loss(pred, label, end_points, inner_label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    print('get_model', pred, label, loss, inner_label)

    classify_loss = 0.0
    bsize = pred.get_shape()[0].value
    for b in range(bsize):
        inIdx = tf.where(inner_label[b,:]>0)
        item_loss = loss[b,:]
        item_inner_loss = tf.gather_nd(item_loss,inIdx)
        classify_loss += tf.cond(tf.equal(tf.size(item_inner_loss), 0), lambda: 0.0, \
                                 lambda:tf.reduce_mean(item_inner_loss))

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss