from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf
import skimage
import scipy.io as sio

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
tf.app.flags.DEFINE_string('modelname', 'baseline', None)

tf.app.flags.DEFINE_integer('max_iter', 700000, None)
tf.app.flags.DEFINE_integer('snapshot_interval', 100000, None)
tf.app.flags.DEFINE_integer('batch_size', 1, None)
tf.app.flags.DEFINE_integer('H', 320, None)
tf.app.flags.DEFINE_integer('W', 320, None)
tf.app.flags.DEFINE_integer('num_steps', 20, None)
tf.app.flags.DEFINE_integer('vf_h', 40, None)
tf.app.flags.DEFINE_integer('vf_w', 40, None)
tf.app.flags.DEFINE_integer('vf_dim', 2048, None)

tf.app.flags.DEFINE_boolean('dcrf', False, None)

def train(reader, snapshot_file, visual_feat_dir):
    model = Model(
        'train',
        vocab_size,
        H=FLAGS.H,
        W=FLAGS.W,
        vf_h=FLAGS.vf_h,
        vf_w=FLAGS.vf_w,
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
            im_name = str(batch['im_name_batch'])
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

            visual_feat = np.load(visual_feat_dir + im_name + '.npz')['arr_0']

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
        acc_all_avg = avg_decay * acc_all_avg + (1 - avg_decay) * acc_all
        acc_pos_avg = avg_decay * acc_pos_avg + (1 - avg_decay) * acc_pos
        acc_neg_avg = avg_decay * acc_neg_avg + (1 - avg_decay) * acc_neg

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

def test(reader, snapshot_file, visual_feat_dir):
    model = Model(
        mode='test',
        vocab_size=vocab_size,
        H=FLAGS.H,
        W=FLAGS.W,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.num_steps)

    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I = cum_U = cum_I_dcrf = cum_U_dcrf = 0
    seg_total = 0
    seg_correct = [0 for _ in range(len(eval_seg_iou_list))]
    if FLAGS.dcrf: seg_correct_dcrf = [0 for _ in range(len(eval_seg_iou_list))]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    snapshot_loader = tf.train.Saver()
    snapshot_loader.restore(sess, snapshot_file % (FLAGS.max_iter))

    for n_iter in range(reader.num_batch):
        sys.stdout.write('Testing %d/%d\r' % (n_iter + 1, reader.num_batch))
        sys.stdout.flush()

        batch = reader.read_batch(is_log=False)
        text = batch['text_batch']
        im_name = str(batch['im_name_batch'])
        mask = batch['mask_batch'].astype(np.float32)
        sent_id = batch['sent_id']

        visual_feat = np.load(visual_feat_dir + im_name + '.npz')['arr_0']

        score_val, pred_val, sigm_val = sess.run(
            [model.score, model.pred, model.sigm],
            feed_dict={
                model.words: np.expand_dims(text, axis=0),
                model.visual_feat: visual_feat
            })

        pred_val = np.squeeze(pred_val)
        pred_raw = (pred_val >= score_thresh).astype(np.float32)
        predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])

        I, U = eval_tools.compute_mask_IU(predicts, mask)
        cum_I += I
        cum_U += U
        for n_eval_iou in range(len(eval_seg_iou_list)):
            seg_correct[n_eval_iou] += (I/U >= eval_seg_iou_list[n_eval_iou])

        if FLAGS.dcrf:
            sigm_val = np.squeeze(sigm_val)
            d = densecrf.DenseCRF2D(FLAGS.W, FLAGS.H, 2)
            U = np.expand_dims(-np.log(sigm_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=im, compat=10)
            Q = d.inference(5)
            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((FLAGS.H, FLAGS.W)).astype(np.float32)
            predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])

            I, U = eval_tools.compute_mask_IU(predicts, mask)
            cum_I_dcrf += I
            cum_U_dcrf += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                seg_correct_dcrf[n_eval_iou] += (I/U >= eval_seg_iou_list[n_eval_iou])

        seg_total += 1

        sio.savemat('./results/%d.mat' % sent_id, {'mask': predicts.astype(np.bool), 'iou': I / U}, do_compression=True)

    msg = 'cumulative IoU = %f' % (cum_I/cum_U)
    if FLAGS.dcrf:
        msg += '\tcumulative IoU (dcrf) = %f' % (cum_I_dcrf/cum_U_dcrf)
    print(msg)


def main(argv):
    # Variable parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if FLAGS.vf_h == 64 and FLAGS.vf_w == 64:
        data_folder = '../quadsplit/data/' + FLAGS.dataset + '/' + FLAGS.setname + '_batch'
    else:
        data_folder = './data/' + FLAGS.dataset + '/' + FLAGS.setname + '_batch'
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
    if FLAGS.vf_h == 64 and FLAGS.vf_w == 64:
        visual_feat_dir = visual_feat_dir[:-1] + '_512x512/'

    if FLAGS.mode == 'train':
        if not os.path.isdir(FLAGS.sfolder): os.makedirs(FLAGS.sfolder)
        train(reader, snapshot_file, visual_feat_dir)
    elif FLAGS.mode == 'test':
        test(reader, snapshot_file, visual_feat_dir)
    else:
        raise ValueError('Invalid mode: %s' % FLAGS.mode)

if __name__ == '__main__':
    # Fixed parameters
    vocab_size = 8803 if FLAGS.dataset == 'referit' else 12112
    tf.app.run()    # parse command line arguments
