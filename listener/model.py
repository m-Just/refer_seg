import numpy as np
import tensorflow as tf

from util import loss
from util.processing_tools import *

class Listener_model(object):
    def __init__(self,
        mode,
        vocab_size,
        H = 320,
        W = 320,
        batch_size = 1,
        rnn_size = 1000,
        num_steps = 20,
        vf_h = 40,
        vf_w = 40,
        vf_dim = 2048,
        v_emb_dim = 1000,
        w_emb_dim = 1000,
        mlp_dim = 500,
        start_lr = 2.5e-4,
        end_lr = 1e-5,
        lr_decay_step = 700000,
        lr_decay_rate = 1.0,
        weight_decay = 5e-4):

        # Task parameters
        self.mode = mode
        self.vocab_size = vocab_size

        # Hyper parameters
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.num_steps = num_steps
        self.H = H
        self.W = W
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.vf_dim = vf_dim
        self.v_emb_dim = v_emb_dim
        self.w_emb_dim = w_emb_dim
        self.mlp_dim = mlp_dim
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay

        # Placeholders
        self.words = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.visual_feat = tf.placeholder(tf.float32, [self.batch_size, self.vf_h, self.vf_w, self.vf_dim])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])

        # Build model
        with tf.variable_scope('refer_seg'):
            self.build_graph()
            if self.mode == 'train': self.train_op()

    def build_graph(self):
        # Obtain visual feature
        visual_feat = self._conv('mlp0', self.visual_feat, 1, self.vf_dim,
            self.v_emb_dim, [1, 1, 1, 1])

        # word embedding
        embed_mat = tf.get_variable('embedding', [self.vocab_size, self.w_emb_dim],
            initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
        embed_seq = tf.nn.embedding_lookup(embed_mat, tf.transpose(self.words))

        # LSTM cell for language feature extraction
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        state = lstm_cell.zero_state(self.batch_size, tf.float32)

        def skip():
            return tf.constant(0.), state

        def update_cell():
            return lstm_cell(embed_seq[n, :, :], state)

        with tf.variable_scope('RNN'):
            for n in range(self.num_steps):
                if n > 0: tf.get_variable_scope().reuse_variables()
                rnn_output, state = tf.cond(
                    tf.equal(self.words[0, n], tf.constant(0)),
                    skip, update_cell)

        # Obtain language feature
        lang_feat = tf.reshape(rnn_output, [self.batch_size, 1, 1, self.rnn_size])
        lang_feat = tf.nn.l2_normalize(lang_feat, 3)
        lang_feat = tf.tile(lang_feat, [1, self.vf_h, self.vf_w, 1])

        # Generate spatial grid
        spatial_feat = tf.convert_to_tensor(generate_spatial_batch(
            self.batch_size, self.vf_h, self.vf_w))

        # Fuse all features
        feat_all = tf.concat([visual_feat, lang_feat, spatial_feat], 3)

        fusion = self._conv('fusion', feat_all, 1,
            self.v_emb_dim + self.rnn_size + 8, self.mlp_dim, [1, 1, 1, 1])
        fusion = tf.nn.relu(fusion)

        # Obtain score and prediction
        self.score = self._conv('conv_cls', fusion, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.pred = tf.image.resize_bilinear(self.score, [self.H, self.W])
        if self.mode == 'test':
            self.sigm = tf.sigmoid(self.pred)

        # Listener component
        visual_feat_flat = tf.layers.flatten(visual_feat)
        visual_mlp0 = tf.contrib.layers.fully_connected(visual_feat_flat, 1000, activation_fn=tf.nn.relu)
        visual_mlp1 = tf.contrib.layers.fully_connected(visual_mlp0, 500, activation_fn=None)
        self.visual_embed = tf.nn.l2_normalize(visual_mlp1, dim=1)

        lang_mlp0 = tf.contrib.layers.fully_connected(rnn_output, 1000, activation_fn=tf.nn.relu)
        lang_mlp1 = tf.contrib.layers.fully_connected(lang_mlp0, 500, activation_fn=None)
        self.lang_embed = tf.nn.l2_normalize(lang_mlp1, dim=1)

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

        # Define loss
        self.target_coarse = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        self.cls_loss = loss.weighed_logistic_loss(self.pred, self.target_fine, 1, 1)
        self.reg_loss = loss.l2_regularization_loss(rvars, self.weight_decay)
        self.cosine_similarity = tf.reduce_sum(tf.multiply(self.visual_embed, self.lang_embed), axis=1)
        self.emb_loss = tf.maximum(0, tf.constant(1.0) - self.cosine_similarity)
        self.sum_loss = self.cls_loss + self.reg_loss + self.emb_loss

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
