import json
import skimage
import skimage.io
import skimage.transform

import numpy as np
import tensorflow as tf

def resize(h, w):
    im_list = []
    with open('/data/ryli/kcli/visual_genome/attr_train.json') as f:
        im_list.extend([im['image_id'] for im in json.load(f)])

    channel_mean = np.zeros([3], dtype=np.float32)

    im_dir = '/data/ryli/kcli/visual_genome/VG_100K'
    save_dir = './images'
    for im_id in im_list:
        im = skimage.io.imread('%s/%d.jpg' % (im_dir, im_id))
        im = skimage.transform.resize(im, (h, w))
        channel_mean += np.mean(im, axis=(0, 1))
        skimage.io.imsave('%s/%d.jpg' % (save_dir, im_id), im)
    print 'total %d images processed' % len(im_list)

    channel_mean /= float(len(im_list))
    channel_mean *= 255.0
    print channel_mean

resize(384, 384)
