import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

w_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(0.1)

init_params = {
    "kernel_initializer": w_init,
    "bias_initializer": b_init,
}

def Generator(x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(x, 512, activation=tf.nn.relu, **init_params)
        h2 = tf.layers.dense(h1, 256, activation=tf.nn.relu, **init_params)
        h3 = tf.layers.dense(h2, 512, activation=tf.nn.relu, **init_params)
        x_out = tf.layers.dense(h3, 784, activation=tf.nn.sigmoid, **init_params)
        return x_out

def Discriminator(x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(x, 512, activation=tf.nn.relu, **init_params)
        d_logit = tf.layers.dense(h1, 1, activation=None, **init_params)
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob, d_logit

def plot(samples, size, name, figsize=(4,4)):
    size = int(size)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(figsize[0], figsize[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(size, size), cmap='Greys_r')

    plt.savefig('{}.png'.format(name), bbox_inches='tight')
    plt.close(fig)

x1_ = tf.placeholder(tf.float32, shape=[None, 784])
x2_ = tf.placeholder(tf.float32, shape=[None, 784])

x2_enc = Generator(x1_, 'G1', False)
x1_rec = Generator(x2_enc, 'G2', False)
d2_fake_prob, d2_fake_logit = Discriminator(x2_enc, 'D2', False)
d2_real_prob, d2_real_logit = Discriminator(x2_, 'D2', True)

x1_enc = Generator(x2_, 'G2', True)
x2_rec = Generator(x1_enc, 'G1', True)
d1_fake_prob, d1_fake_logit = Discriminator(x1_enc, 'D1', False)
d1_real_prob, d1_real_logit = Discriminator(x1_, 'D1', True)

g1_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G1')
g2_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G2')
d1_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D1')
d2_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D2')

rec1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x1_ - x1_rec), 1))
g2_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d2_fake_logit), logits=d2_fake_logit))
d2_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(d2_fake_logit), logits=d2_fake_logit))
d2_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d2_real_logit), logits=d2_real_logit))
d2_loss = d2_loss_real + d2_loss_fake

rec2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x2_ - x2_rec), 1))
g1_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d1_fake_logit), logits=d1_fake_logit))
d1_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(d1_fake_logit), logits=d1_fake_logit))
d1_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d1_real_logit), logits=d1_real_logit))
d1_loss = d1_loss_real + d1_loss_fake

g1_solver = tf.train.AdamOptimizer(1e-3).minimize(rec1_loss + g2_loss, var_list=g1_var+g2_var)
g2_solver = tf.train.AdamOptimizer(1e-3).minimize(rec2_loss + g1_loss, var_list=g1_var+g2_var)
d1_solver = tf.train.AdamOptimizer(1e-3).minimize(d1_loss, var_list=d1_var)
d2_solver = tf.train.AdamOptimizer(1e-3).minimize(d2_loss, var_list=d2_var)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

out_folder = 'out_cyclegan2/'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
i = 0

batch_size = 64
for it in range(20001):
    x1_batch, _ = mnist.train.next_batch(batch_size)
    x2_batch = 1. - mnist.train.next_batch(batch_size)[0]

    g1_loss_ = g2_loss_ = d1_loss_ = d2_loss_ = 0.
    _, d1_loss_ = sess.run([d1_solver, d1_loss], feed_dict={x1_:x1_batch, x2_: x2_batch})
    _, d2_loss_ = sess.run([d2_solver, d2_loss], feed_dict={x1_:x1_batch, x2_: x2_batch})
    _, g1_loss_ = sess.run([g1_solver, g1_loss], feed_dict={x1_:x1_batch, x2_: x2_batch}) 
    _, g2_loss_ = sess.run([g2_solver, g2_loss], feed_dict={x1_:x1_batch, x2_: x2_batch}) 
    
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('G1_loss: {:.4}, D1_loss: {:.4}'.format(g1_loss_, d1_loss_))
        print('G2_loss: {:.4}, D2_loss: {:.4}'.format(g2_loss_, d2_loss_))
        print()

        x2_enc_, x1_rec_ = sess.run([x2_enc, x1_rec], feed_dict={x1_:x1_batch[:6]})
        x1_enc_, x2_rec_ = sess.run([x1_enc, x2_rec], feed_dict={x2_:x2_batch[:6]})
        out_img = np.concatenate([x1_batch[:6], x2_enc_, x1_rec_, x2_batch[:6], x1_enc_, x2_rec_], 0)
        fig = plot(out_img, 28, out_folder+str(i).zfill(4), figsize=(6,6))
        i += 1
