import numpy as np
import tensorflow as tf

from util import loss

class Skip_thoughts_speaker_model(object):
    def __init__(self,
        mode,
        batch_size=1,
        H=320,
        W=320,
        vf_h=40,
        vf_w=40,
        vf_dim=2048,
        v_emb_dim=1024,
        fusion_dim=2048,
        enc_dim=2400,
        start_lr = 2.5e-4,
        end_lr = 1e-5,
        lr_decay_step=700000,
        lr_decay_rate = 1.0,
        weight_decay = 5e-4):
        self.mode = mode
        self.batch_size = batch_size
        self.H = H
        self.W = W
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.vf_dim = vf_dim
        self.v_emb_dim = v_emb_dim
        self.fusion_dim = fusion_dim
        self.enc_dim = enc_dim
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay

        with tf.variable_scope('refer_seg'):
            self.build_graph()
            if self.mode == 'train': self.train_op()

    def build_graph(self):
        # Placeholders
        self.visual_feat = tf.placeholder(tf.float32, [self.batch_size, self.vf_h, self.vf_w, self.vf_dim])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])
        self.encoding = tf.placeholder(tf.float32, [self.batch_size, self.enc_dim])

        visual_embed = self._conv('visual_embed', self.visual_feat, 1, self.vf_dim,
            self.v_emb_dim, [1, 1, 1, 1])
        spatial_feat = tf.convert_to_tensor(generate_spatial_batch(
            self.batch_size, self.vf_h, self.vf_w))
        feat_all = tf.concat([visual_embed, spatial_feat], 3)

        fusion = self._conv('fusion', feat_all, 1, self.v_emb_dim + 8,
            self.fusion_dim, [1, 1, 1, 1])
        fusion = tf.nn.relu(fusion)

        self.score = self._conv('conv_cls', fusion, 3, self.fusion_dim, self.enc_dim, [1, 1, 1, 1])
        self.pred = tf.image.resize_bilinear(self.score, [self.H, self.W])

    def train_op(self):
        # Collect variables for training
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('refer_seg')]
        print('Collecting variables for training:')
        for var in tvars: print('\t%s' % var.name)
        print('Done.')

        # Collect variables for regularization
        rvars = [var for var in tf.trainable_variables() if var.op.name.startswith('refer_seg')]
        print('Collecting variables for regularization:')
        for var in rvars: print('\t%s' % var.name)
        print('Done.')

        self.target_coarse = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        self.masked_score = tf.gather_nd(self.score, tf.where(self.target_coarse > 0.5))
        self.mean_score = tf.reduce_mean(self.masked_score, axis=1)

        self.spk_loss = tf.reduce_mean(tf.squared_difference(self.mean_score, self.encoding))
        self.reg_loss = loss.l2_regularization_loss(rvars, self.weight_decay)
        self.sum_loss = self.spk_loss + self.reg_loss

        # Define learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, self.global_step,
            self.lr_decay_step, self.end_lr, self.lr_decay_rate)

        # Define optimization process
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.sum_loss, var_list=tvars)

        var_lr_mult = {}
        for var in tvars:
            var_lr_mult[var] = 2.0 if var.op.name.find('biases') > 0 else 1.0
        print('Setting variable learning rate multiplication:')
        for var in tvars:
            print('\t%s: %f' % (var.name, var_lr_mult[var]))
        print('Done.')

        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in grads_and_vars]
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b
