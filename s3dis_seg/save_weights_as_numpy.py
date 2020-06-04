import argparse
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
sys.path.append(baseDir)
sys.path.append(os.path.join(rootDir, 'models'))
sys.path.append(os.path.join(rootDir, 'utils'))
import data_util
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='SPH3D_s3dis_resnet')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--test_area', type=int, default=5, help='which Area is the test fold')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_name', default='model.ckpt', help='model checkpoint file path [default: model.ckpt]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MODEL_NAME = FLAGS.model_name
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  # import network module

LOG_DIR = os.path.join(rootDir,'log_s3dis_Area_%d_resEncBlock'%(FLAGS.test_area))

spec = importlib.util.spec_from_file_location('', os.path.join(LOG_DIR, FLAGS.model+'.py'))
MODEL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODEL)

if not os.path.exists(os.path.join(LOG_DIR, 'weights')):
    os.mkdir(os.path.join(LOG_DIR, 'weights'))

spec = importlib.util.spec_from_file_location('', os.path.join(LOG_DIR, 's3dis_config_resnet.py'))
net_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net_config)

NUM_POINT = net_config.num_input
NUM_CLASSES = 13
INPUT_DIM = 6
HOSTNAME = socket.gethostname()

def placeholder_inputs(batch_size, num_point):
    input_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, INPUT_DIM))
    label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    inner_label_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return input_pl, label_pl, inner_label_pl

# def placeholder_inputs(batch_size, num_point, normal=False):
#     xyz_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
#     label_pl = tf.placeholder(tf.int32, shape=(batch_size))
#     if normal:
#         normal_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
#     else:
#         normal_pl = None
#
#     return xyz_pl, normal_pl, label_pl


def evaluate():
    # =================================Define the Graph================================
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # =================================Define the Graph================================
        input_pl, label_pl, inner_label_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        training_pl = tf.placeholder(tf.bool, shape=())
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get model and loss
        pred, end_points = MODEL.get_model(input_pl, training_pl, config=net_config)
        MODEL.get_loss(pred, label_pl, end_points, inner_label_pl)
        if net_config.weight_decay is not None:
            reg_loss = tf.multiply(tf.losses.get_regularization_loss(), net_config.weight_decay, name='reg_loss')
            tf.add_to_collection('losses', reg_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    # =====================================The End=====================================

    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    var = [v for v in tf.trainable_variables() if not 'bn' in v.name]
    print(var)

    with tf.Session(config=config) as sess:
        # Load the model
        saver.restore(sess, os.path.join(LOG_DIR, MODEL_NAME))
        print("Model restored.")

        for v in var:
            W = sess.run(v)
            names = v.name.split('/')
            names[-1]= names[-1][0:-2]
            fname = '_'.join(names)
            if 'depthwise' in fname:
                # np.savetxt(os.path.join(LOG_DIR, 'weights', fname+'.csv'), W, delimiter=",")
                sio.savemat(os.path.join(LOG_DIR, 'weights', fname+'.mat'), {'weights':W})
    # =====================================The End=====================================



if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()