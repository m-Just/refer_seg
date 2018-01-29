from glob import glob

import numpy as np
import tensorflow as tf
import skimage

from util import im_processing
from deeplab_resnet import model as deeplab101

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', None, None)
tf.app.flags.DEFINE_string('dataset', None, None)
tf.app.flags.DEFINE_string('setname', None, None)

def main(argv):
    mu = np.array((104.00698793, 116.66876762, 122.67891434))
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    data_folder = '../data/' + FLAGS.dataset + '/' + FLAGS.setname + '_visual_feat/'
    data_prefix = FLAGS.dataset + '_' + FLAGS.setname
    if FLAGS.dataset == 'referit':
        im_dir = '/data/ryli/text_objseg/exp-referit/referit-dataset/images/'
    elif FLAGS.dataset = 'coco':
        im_dir = '/data/ryli/datasets/coco/images/train2014/'
    else:
        print('dataset must be referit or coco')
        return

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    input = tf.placeholder(tf.float32, [1, 320, 320, 3])
    resmodel = deeplab101.DeepLabResNetModel({'data': input}, is_training=False)

    visual_feat = resmodel.layers['res5c_relu']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    imgs = glob(im_dir + '*.jpg')
    for i, im_path in enumerate(imgs):
        print('saving visual features %d / %d' % (i + 1, len(imgs)))
        im = skimage.io.imread(im_path)
        im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, 320, 320))
        im = im.astype(np.float32)[:, :, ::-1] - mu

        vis_feat_val = sess.run(visual_feat, feed_dict={input: im})

        save_path = data_folder + im_path.split('/')[-1][:-4] + '.npy'
        np.save(save_path, vis_feat_val)

    print 'visual feature extraction complete'

if __name__ == '__main__':
    tf.app.run()    # parse command line arguments
