import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

from model import Detector

d_scale = 16
H = W = 24 * d_scale
detector = Detector('train', 1, H, W, d_scale)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

a = [0 for _ in range(1024)]
a[0] = 1
a[21] = 1
a[999] = 1
test_attr0 = [[a for _ in range(2)]]
_, m, l, e0 = sess.run([detector.train_step, detector.bbox_cls_pred, detector.cls_loss, detector.end_points['conv_cls']],
    feed_dict={detector.im: np.ones((1, H, W, 3), dtype=np.float32),
               detector.bbox: [[[0,0,50,50], [34,10,100,62]]],
               detector.attr: test_attr0})
print m.shape
print l
print e0[0, 0, 0]

a = [0 for _ in range(1024)]
a[20] = 1
a[121] = 1
test_attr1 = [[a for _ in range(3)]]
_, m, l, e1 = sess.run([detector.train_step, detector.bbox_cls_pred, detector.cls_loss, detector.end_points['conv_cls']],
    feed_dict={detector.im: np.ones((1, H, W, 3), dtype=np.float32),
               detector.bbox: [[[0,0,50,50], [34,10,100,62], [2,4,60,100]]],
               detector.attr: test_attr1})
print m.shape
print l
print e1[0, 0, 0]

print (e0 == e1).all()
