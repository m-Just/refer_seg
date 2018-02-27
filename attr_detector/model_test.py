from itertools import permutations

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

class Detector(object):
    def __init__(self, mode, batch_size, H, W, attr_num, attr_freq, init_lr, weight_decay=1e-4):
        # conv classification layer parameters
        self.d_scale = 16
        self.feat_size = (24, 24)
        assert H == self.feat_size[0] * self.d_scale
        assert W == self.feat_size[1] * self.d_scale

        # model parameters
        self.mode = mode
        self.batch_size = batch_size
        self.H = H
        self.W = W
        self.attr_num = attr_num

        self.init_lr = init_lr
        self.weight_decay = weight_decay

        # placeholders
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.bbox = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.attr = tf.placeholder(tf.float32, [self.batch_size, self.attr_num])

        self.attr_freq = tf.constant(attr_freq, dtype=tf.float32)

        # graph building
        self.end_points = {}
        with tf.variable_scope('attr_detector'):
            self.build_graph()
            if mode == 'train': self.train_op()

    def build_graph(self):
        # Original VGG-16 blocks
        net = slim.repeat(self.im, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        self.end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        self.end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        self.end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        self.end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        #self.end_points['block5'] = net
        #net= slim.max_pool2d(net, [3, 3], stride=1, paddding='SAME', scope='pool5')

        # Additional detection blocks
        net = slim.conv2d(net, 1024, [3, 3], scope='conv6')
        self.end_points['block6'] = net
        net = tf.layers.dropout(net, rate=0.5, training=(self.mode == 'train'))

        #TODO: add bounding box size and shape as additional feature
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        self.end_points['block7'] = net
        net = tf.layers.dropout(net, rate=0.5, training=(self.mode == 'train'))

        cls_pred = slim.conv2d(net, self.attr_num, [3, 3], activation_fn=None, scope='conv_cls')
        self.end_points['conv_cls'] = cls_pred

        # Extract classification score for each bounding box
        feat_bbox = tf.cast(tf.truediv(self.bbox, float(self.d_scale)) + 0.5, tf.int32)

        all_pred = []
        for i in range(self.batch_size):
            x, y, w, h = [feat_bbox[i, _] for _ in range(4)]
            x = tf.minimum(x, self.W / self.d_scale - 2)
            y = tf.minimum(y, self.H / self.d_scale - 2)
            w = tf.maximum(w, 1)
            h = tf.maximum(h, 1)
            region = tf.slice(cls_pred[i], [y, x, 0], [h, w, -1])
            ith_pred = tf.reduce_sum(region, axis=[0, 1]) / tf.cast(h * w, tf.float32)
            all_pred.append(ith_pred)

        self.bbox_cls_pred = tf.stack(all_pred)

    def train_op(self):
        # Running eval metric
        precision = tf.constant(0, dtype=tf.float32)
        recall = tf.constant(0, dtype=tf.float32)
        top_recall = tf.constant(0, dtype=tf.float32)

        for i in range(self.batch_size):
            true_num = tf.reduce_sum(self.attr, axis=-1)[i]
            score, indices = tf.nn.top_k(self.bbox_cls_pred[i], k=tf.cast(true_num, tf.int32))
            top_tp = tf.reduce_sum(tf.gather(self.attr[i], indices))

            positive = tf.where(self.bbox_cls_pred[i] > 0.0, tf.ones([self.attr_num]), tf.zeros([self.attr_num]))
            pos_num = tf.reduce_sum(positive)
            tp = tf.reduce_sum(positive * self.attr[i])

            precision += tf.cond(pos_num > 0, lambda: tp / pos_num, lambda: 0.0)
            recall += tp / true_num
            top_recall += top_tp / true_num

        self.precision = precision / tf.cast(self.batch_size, tf.float32)
        self.recall = recall / tf.cast(self.batch_size, tf.float32)
        self.top_recall = top_recall / tf.cast(self.batch_size, tf.float32)

        #self.cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.attr, logits=self.bbox_cls_pred)
        self.cls_loss = self.weighted_cross_entropy_with_logits(self.attr,
            self.bbox_cls_pred, self.attr_freq / 1.2)
        self.cls_loss *= 0.1 * tf.reduce_max(self.attr_freq) / self.attr_freq
        self.cls_loss = tf.reduce_mean(tf.reduce_sum(self.cls_loss, axis=1))

        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
        reg_var = [var for var in tf.trainable_variables() if var.op.name.find('weights') > 0]
        self.reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=reg_var)

        self.sum_loss = self.cls_loss + self.reg_loss

        # Define optimization process
        #TODO: learning rate decay
        self.learning_rate = tf.constant(self.init_lr)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.sum_loss, var_list=tf.trainable_variables())
        #TODO: double bias gradient

        self.global_step = tf.Variable(0, trainable=False)
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def weighted_cross_entropy_with_logits(labels, logits, freq):
        pos_weight = 1 + (1 / freq - 2) * labels
        return (1 - labels) * logits  + pos_weight * \
            (tf.log(1 + tf.exp(-tf.abs(logits))) + tf.reduce_max(-logits, 0))
