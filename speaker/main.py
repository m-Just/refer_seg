import numpy as np
import tensorflow as tf

from model import Skip_thoughts_speaker_model as Model

# Command line arguments
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', None, None)
tf.app.flags.DEFINE_string('mode', None, None)
tf.app.flags.DEFINE_string('dataset', None, None)
tf.app.flags.DEFINE_string('setname', None, None)
tf.app.flags.DEFINE_string('sfolder', 'ckpts', None)
tf.app.flags.DEFINE_string('modelname', 'speaker', None)

tf.app.flags.DEFINE_integer('max_iter', 700000, None)
tf.app.flags.DEFINE_integer('snapshot_interval', 100000, None)
tf.app.flags.DEFINE_integer('batch_size', 1, None)
tf.app.flags.DEFINE_integer('H', 320, None)
tf.app.flags.DEFINE_integer('W', 320, None)
tf.app.flags.DEFINE_integer('vf_h', 40, None)
tf.app.flags.DEFINE_integer('vf_w', 40, None)
tf.app.flags.DEFINE_integer('vf_dim', 2048, None)
tf.app.flags.DEFINE_integer('enc_dim', 2400, None)

tf.app.flags.DEFINE_boolean('dcrf', False, None)

def train(reader, snapshot_file, visual_feat_dir):
    model = Model(
        'train',
        vocab_size,
        H=FLAGS.H,
        W=FLAGS.W,
        batch_size=FLAGS.batch_size,
        lr_decay_step=FLAGS.max_iter)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    snapshot_saver = tf.train.Saver(max_to_keep=1000)

    visual_feat_batch = np.zeros((FLAGS.batch_size, FLAGS.vf_h, FLAGS.vf_w, FLAGS.vf_dim), dtype=np.float32)
    mask_batch = np.zeros((FLAGS.batch_size, FLAGS.H, FLAGS.W, 1), dtype=np.float32)
    encoding_batch = np.zeros((FLAGS.batch_size, FLAGS.enc_dim), dtype=np.float32)

    acc_all_avg = acc_pos_avg = acc_neg_avg = spk_loss_avg = 0.0
    avg_decay = 0.99

    iters_per_log = 100
    for n_iter in range(FLAGS.max_iter):
        for n_batch in range(FLAGS.batch_size):
            batch = reader.read_batch(is_log=(n_batch ==0 and n_iter % iters_per_log == 0))
            im_name = str(batch['im_name_batch'])
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)
            encoding = batch['encoding_batch'].astype(np.float32)

            visual_feat = np.load(visual_feat_dir + im_name + '.npz')['arr_0']

            visual_feat_batch[n_batch, ...] = visual_feat
            mask_batch[n_batch, ...] = mask
            encoding_batch[n_batch, ...] = encoding

        _, spk_loss_val, lr_val, score_val, label_coarse_val = sess.run(
            [model.train_step, model.spk_loss, model.learning_rate,
             model.score, model.target_coarse],
            feed_dict={
                model.visual_feat: visual_feat_batch,
                model.target_fine: mask_batch,
                model.encoding: encoding_batch
            })

        spk_loss_avg = avg_decay * spk_loss_avg + (1 - avg_decay) * spk_loss_val

        if n_iter % iters_per_log == 0:
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                % (n_iter, spk_loss_val, spk_loss_avg, lr_val))

        if (n_iter + 1) % FLAGS.snapshot_interval == 0 or (n_iter + 1) >= FLAGS.max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter + 1))
            print('Snapshot saved to' + snapshot_file % (n_iter + 1))

    print('Optimization done.')
