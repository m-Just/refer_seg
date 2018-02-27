import os
import json
import skimage
import skimage.io, skimage.transform

import numpy as np
import tensorflow as tf

from model import Detector

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
tf.app.flags.DEFINE_float('init_lr', 1e-5, None)
tf.app.flags.DEFINE_float('final_lr', 1e-5, None)

def train():
    # load model graph
    detector = Detector('train', FLAGS.batch_size, FLAGS.H, FLAGS.W, FLAGS.attr_num) 

    # load training data
    with open(FLAGS.train_data_list % FLAGS.attr_num) as f:
        img_list = json.load(f)
    img_num = len(img_list)
    channel_mean = [119.41751862, 114.7717514, 105.80799103]

    attr_list = list()
    with open(FLAGS.attr_label % FLAGS.attr_num) as f:
        attr_list.extend([line.strip() for line in f])
    assert len(attr_list) == FLAGS.attr_num
    attr_ind = dict([(a, i) for i, a in enumerate(attr_list)])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    snapshot_saver = tf.train.Saver(max_to_keep=1000)

    max_epoch = 10
    max_iter = img_num * max_epoch / FLAGS.batch_size
    prec_avg = recl_avg = top_recl_avg = cls_loss_avg = 0.0
    decay = 0.99
    for iter in range(max_iter):
        im_index = iter % img_num
        im_id = img_list[im_index]['image_id']
        attrs = img_list[im_index]['attributes']
        im_w = img_list[im_index]['width']
        im_h = img_list[im_index]['height']
        bbox_num = len(attrs)

        im_batch = np.zeros([FLAGS.batch_size, FLAGS.H, FLAGS.W, 3], dtype=np.float32)
        bbox_batch = np.zeros([FLAGS.batch_size, bbox_num, 4], dtype=np.float32)
        attr_batch = np.zeros([FLAGS.batch_size, bbox_num, FLAGS.attr_num], dtype=np.float32)
        for n_batch in range(FLAGS.batch_size):
            im = skimage.io.imread('%s/%s.jpg' % (FLAGS.im_dir, im_id)).astype(np.float32)
            if len(im.shape) == 2: im = np.stack([im] * 3, axis=-1)
            im -= channel_mean

            x_ratio = float(FLAGS.W) / im_w
            y_ratio = float(FLAGS.H) / im_h
            im_batch[n_batch, ...] = im

            for n_bbox in range(bbox_num):
                a = attrs[n_bbox]
                x = a['x'] * x_ratio
                y = a['y'] * y_ratio
                w = a['w'] * x_ratio
                h = a['h'] * y_ratio
                bbox_batch[n_batch, n_bbox, ...] = [x, y, w-1, h-1]

                attr = a['attributes']
                ind = map(lambda w: attr_ind[w], attr)
                attr_batch[n_batch, n_bbox, ind] = 1

        _, lr_val, cls_loss_val, prec_val, recl_val, top_recl_val = sess.run(
            [detector.train_step, detector.learning_rate,
             detector.cls_loss, detector.precision, detector.recall, detector.top_recall],
            feed_dict={detector.im: im_batch,
                      detector.bbox: bbox_batch,
                      detector.attr: attr_batch})

        prec_avg = prec_avg * decay + prec_val * (1 - decay)
        recl_avg = recl_avg * decay + recl_val * (1 - decay)
        top_recl_avg = top_recl_avg * decay + top_recl_val * (1 - decay)
        cls_loss_avg = cls_loss_avg * decay + cls_loss_val * (1 - decay)

        if iter % 100 == 0: 
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                % (iter, cls_loss_val, cls_loss_avg, lr_val))
            print('iter = %d, prec (cur) = %f, prec (avg) = %f'
                % (iter, prec_val, prec_avg))
            print('iter = %d, recl (cur) = %f, recl (avg) = %f'
                % (iter, recl_val, recl_avg))
            print('iter = %d, t-rc (cur) = %f, t-rc (avg) = %f'
                % (iter, top_recl_val, top_recl_avg))

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
