import os
import json
import skimage
import skimage.io, skimage.transform

import numpy as np
import tensorflow as tf

from model_test import Detector
from data_reader import DataReader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', None, None)
tf.app.flags.DEFINE_string('mode', None, None)
tf.app.flags.DEFINE_integer('max_epoch', 10, None)

tf.app.flags.DEFINE_string('im_dir', './images', None)
tf.app.flags.DEFINE_string('train_data_list',
    '/data/ryli/kcli/visual_genome/attr_top%d_train.json', None)
tf.app.flags.DEFINE_string('attr_label',
    '/data/ryli/kcli/visual_genome/top_%d_attrs.txt', None)

tf.app.flags.DEFINE_integer('batch_size', 1, None)
tf.app.flags.DEFINE_integer('H', 384, None)
tf.app.flags.DEFINE_integer('W', 384, None)
tf.app.flags.DEFINE_integer('attr_num', 1024, None)

tf.app.flags.DEFINE_float('pos_weight', 1.0, None)
tf.app.flags.DEFINE_float('init_lr', 1e-4, None)
tf.app.flags.DEFINE_float('final_lr', 1e-4, None)

def train():
    # load attributes name and their occurrence number
    attr_list = list()
    with open(FLAGS.attr_label % FLAGS.attr_num) as f:
        name, _ = zip(*[line.split('\t') for line in f])
        attr_list.extend([str(n).strip() for n in name])
    assert len(attr_list) == FLAGS.attr_num
    attr_ind = dict([(a, i) for i, a in enumerate(attr_list)])
    
    # load model graph
    detector = Detector('train', FLAGS.batch_size, FLAGS.H, FLAGS.W, FLAGS.attr_num, FLAGS.pos_weight, FLAGS.init_lr) 

    # load training data
    data_reader = DataReader(FLAGS.train_data_list % FLAGS.attr_num, FLAGS.im_dir, FLAGS.batch_size, FLAGS.attr_num)
    channel_mean = [119.41751862, 114.7717514, 105.80799103] # on VG_100K images containing top-1024 attributes

    # start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_saver = tf.train.Saver(max_to_keep=1000)

    prec_avg = recl_avg = top_recl_avg = cls_loss_avg = 0.0
    iter_per_log = 100
    decay = 0.99

    max_iter = data_reader.num_batch * FLAGS.max_epoch
    for iter in range(max_iter):
        is_log = not(iter % iter_per_log)
        batch = data_reader.read_batch(is_log=is_log)
        im_batch = np.array(batch['im'], dtype=np.float32)
        bbox_batch = np.zeros([FLAGS.batch_size, 4], dtype=np.float32)
        attr_batch = np.zeros([FLAGS.batch_size, FLAGS.attr_num], dtype=np.float32)

        im_w = batch['width']
        im_h = batch['height']
        bbox_x = batch['bbox_x']
        bbox_y = batch['bbox_y']
        bbox_w = batch['bbox_w']
        bbox_h = batch['bbox_h']
        attrs = batch['attrs']

        for n_batch in range(FLAGS.batch_size):
            x_ratio = float(FLAGS.W) / im_w[n_batch]
            y_ratio = float(FLAGS.H) / im_h[n_batch]

            x = bbox_x[n_batch] * x_ratio
            y = bbox_y[n_batch] * y_ratio
            w = (bbox_w[n_batch] - 1) * x_ratio
            h = (bbox_h[n_batch] - 1) * y_ratio
            bbox_batch[n_batch, ...] = np.array([x, y, w, h], dtype=np.float32)

            ind = map(lambda w: attr_ind[w], attrs[n_batch])
            attr_batch[n_batch, ind] = 1.0

        _, lr_val, cls_loss_val, prec_val, recl_val, top_recl_val, logits = sess.run(
            [detector.train_step, detector.learning_rate,
             detector.cls_loss, detector.precision, detector.recall, detector.top_recall,
             detector.bbox_cls_pred],
            feed_dict={detector.im: im_batch - channel_mean,
                      detector.bbox: bbox_batch,
                      detector.attr: attr_batch})

        prec_avg = prec_avg * decay + prec_val * (1 - decay)
        recl_avg = recl_avg * decay + recl_val * (1 - decay)
        top_recl_avg = top_recl_avg * decay + top_recl_val * (1 - decay)
        cls_loss_avg = cls_loss_avg * decay + cls_loss_val * (1 - decay)

        if is_log: 
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                % (iter, cls_loss_val, cls_loss_avg, lr_val))
            print('iter = %d, prec (cur) = %f, prec (avg) = %f'
                % (iter, prec_val, prec_avg))
            print('iter = %d, recl (cur) = %f, recl (avg) = %f'
                % (iter, recl_val, recl_avg))
            print('iter = %d, t-rc (cur) = %f, t-rc (avg) = %f'
                % (iter, top_recl_val, top_recl_avg))
            print(logits[0])

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()
    else:
        print 'invalid mode argument %s' % FLAGS.mode

if __name__ == '__main__':
    tf.app.run()
