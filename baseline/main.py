from __future__ import division

import os
import numpy as np
import tensorflow as tf
import skimage

from model import Baseline_model as Model
from pydensecrf import densecrf

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools

# Command line arguments
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', None, None)
tf.app.flags.DEFINE_string('mode', None, None)
tf.app.flags.DEFINE_string('dataset', None, None)
tf.app.flags.DEFINE_string('setname', None, None)
tf.app.flags.DEFINE_string('sfolder', 'ckpts', None)
tf.app.flags.DEFINE_string('modelname', 'convlstm_p543', None)

tf.app.flags.DEFINE_int('max_iter', 700000, None)
tf.app.flags.DEFINE_int('snapshot_interval', 100000, None)
tf.app.flags.DEFINE_int('batch_size', 1, None)
tf.app.flags.DEFINE_int('H', 320, None)
tf.app.flags.DEFINE_int('W', 320, None)
tf.app.flags.DEFINE_int('num_steps', 20, None)

tf.app.flags.DEFINE_boolean('dcrf', False, None)

def train():
    model = Model(
        mode='train',
        vocab_size=vocab_size,
        H=FLAGS.H,
        W=FLAGS.W,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.num_steps,
        lr_decay_step=FLAGS.max_iter)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    snapshot_saver = tf.train.Saver(max_to_keep=1000)

    text_batch = np.zeros((FLAGS.batch_size, FLAGS.num_steps), dtype=np.float32)
    visual_feat_batch = np.zeros((FLAGS.batch_size, FLAGS.vf_h, FLAGS.vf_w, FLAGS.vf_dim), dtype=np.float32)
    mask_batch = np.zeros((FLAGS.batch_size, FLAGS.H, FLAGS.W, 1), dtype=np.float32)

    acc_all_avg = acc_pos_avg = acc_neg_avg = cls_loss_avg = 0.0
    avg_decay = 0.99

    iters_per_log = 100
    for n_iter in range(FLAGS.max_iter):
        for n_batch in range(FLAGS.batch_size):
            batch = reader.read_batch(is_log=(n_batch ==0 and n_iter % iters_per_log == 0))
            text = batch['text_batch']
            im_name = batch['im_name_batch']
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

            visual_feat = np.load(visual_feat_dir + im_name + '.npz')

            text_batch[n_batch, ...] = text
            visual_feat_batch[n_batch, ...] = visual_feat
            mask_batch[n_batch, ...] = mask

        _, cls_loss_val, lr_val, score_val, label_coarse_val = sess.run(
            [model.train_step, model.cls_loss, model.learning_rate,
             model.score, model.target_coarse],
            feed_dict={
                model.words: text_batch,
                model.visual_feat: visual_feat_batch,
                model.target_fine: mask_batch
            })

        cls_loss_avg = avg_decay * cls_loss_avg + (1 - avg_decay) * cls_loss_val
        acc_all, acc_pos, acc_neg = compute_accuracy(score_val, label_coarse_val)
        acc_all_avg = avg_decay * acc_all_avg + (1 - avg_decay) * acc_all_avg
        acc_pos_avg = avg_decay * acc_pos_avg + (1 - avg_decay) * acc_pos_avg
        acc_neg_avg = avg_decay * acc_neg_avg + (1 - avg_decay) * acc_neg_avg

        if n_iter % iters_per_log == 0:
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                % (n_iter, cls_loss_val, cls_loss_avg, lr_val))
            print('iter = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
                % (n_iter, acc_all, acc_pos, acc_neg))
            print('iter = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                % (n_iter, acc_all_avg, acc_pos_avg, acc_neg_avg))

        if (n_iter + 1) % FLAGS.snapshot_interval == 0 or (n_iter + 1) >= FLAGS.max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter + 1))
            print('Snapshot saved to' + snapshot_file % (n_iter + 1))

    print('Optimization done.')

def main(argv):
    # Fixed parameters
    vocab_size = 8803 if FLAGS.dataset == 'referit' else 12112

    # Variable parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    data_folder = '../data/' + FLAGS.dataset + '/' + FLAGS.setname + '_batch'
    data_prefix = FLAGS.dataset + '_' + FLAGS.setname
    reader = data_reader.DataReader(data_folder, data_prefix)
    snapshot_file = os.path.join(FLAGS.sfolder, FLAGS.dataset + '_' +
        FLAGS.modelname + '_iter_%d.tfmodel')

    if FLAGS.dataset in ['unc', 'unc+', 'Gref']:
        visual_feat_dir = '../data/coco/visual_feat/'
    elif FLAGS.dataset == 'referit':
        visual_feat_dir = '../data/referit/visual_feat/'
    else:
        raise ValueError('Unknown dataset %s' % dataset)

    if FLAGS.mode == 'train':
        if not os.path.isdir(FLAGS.sfolder): os.makedirs(FLAGS.sfolder)
        train()
    elif FLAGS.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode: %s' % FLAGS.mode)

if __name__ == '__main__':
    tf.app.run()    # parse command line arguments
