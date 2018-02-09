import time
import os
from glob import glob

import numpy as np
import tensorflow as tf
import skimage.io

from util import im_processing
from deeplab_resnet import model as deeplab101

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', None, None)
tf.app.flags.DEFINE_string('dataset', None, None)

tf.app.flags.DEFINE_integer('H', 320, None)
tf.app.flags.DEFINE_integer('W', 320, None)

tf.app.flags.DEFINE_boolean('compress', False, None)

def main(argv):
    mu = np.array((104.00698793, 116.66876762, 122.67891434))
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    data_folder = '../data/' + FLAGS.dataset + '/visual_feat_%dx%d/' % (FLAGS.H, FLAGS.W)
    if FLAGS.dataset == 'referit':
        im_dir = '/data/ryli/text_objseg/exp-referit/referit-dataset/images/'
    elif FLAGS.dataset == 'coco':
        im_dir = '/data/ryli/datasets/coco/images/train2014/'
    else:
        print('dataset must be referit or coco')
        return

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    input = tf.placeholder(tf.float32, [1, FLAGS.H, FLAGS.W, 3])
    resmodel = deeplab101.DeepLabResNetModel({'data': input}, is_training=False)

    visual_feat = resmodel.layers['res5c_relu']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    pretrained_model = '/data/ryli/text_objseg/tensorflow-deeplab-resnet/models/deeplab_resnet_init.ckpt'
    snapshot_loader = tf.train.Saver()
    snapshot_loader.restore(sess, pretrained_model)

    imgs = glob(im_dir + '*.jpg')
    img_num = len(imgs)
    for i, im_path in enumerate(imgs):
        print('saving visual features %d / %d' % (i + 1, img_num))
        t1 = time.time()
        im = skimage.io.imread(im_path)
        im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, FLAGS.H, FLAGS.W))
        if im.ndim == 2: im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
        im = im.astype(np.float32)[:, :, ::-1] - mu

        vis_feat_val = sess.run(visual_feat, feed_dict={input: np.expand_dims(im, axis=0)})
        t2 = time.time()

        save_path = data_folder + im_path.split('/')[-1][:-4] + '.npz'
        if FLAGS.compress:
            np.savez_compressed(save_path, vis_feat_val)
        else:
            np.savez(save_path, vis_feat_val)
        t3 = time.time()

        np.load(save_path)
        t4 = time.time()
        print('time spent: extraction=%f, saving=%f, loading=%f' % (t2-t1, t3-t2, t4-t3))

    print('visual feature extraction complete')

if __name__ == '__main__':
    tf.app.run()    # parse command line arguments
