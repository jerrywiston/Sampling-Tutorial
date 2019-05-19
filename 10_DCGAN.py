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

x_dim = 784
z_dim = 128

def Generator(z, name, reuse):
    global x_dim, z_dim
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(z, 7*7*64, \
            activation=tf.nn.relu, **init_params)
        h1_re = tf.reshape(h1, [-1,7,7,64])

        h_conv2 = tf.layers.conv2d_transpose(h1_re, 32, [3, 3], \
            activation=tf.nn.relu, strides=[2, 2], \
            padding='same', **init_params)

        h_conv3 = tf.layers.conv2d_transpose(h_conv2, 1, [5, 5], \
            activation=tf.nn.sigmoid, strides=[2, 2], \
            padding='same', **init_params)
        
        x_out = tf.reshape(h_conv3, [-1,784])
        return x_out

def Discriminator(x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        x_re = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.layers.conv2d(x_re, 32, [5, 5], \
            activation=tf.nn.relu, strides=[2, 2], \
            padding='same', **init_params)

        h_conv2 = tf.layers.conv2d(h_conv1, 64, [3, 3], \
            activation=tf.nn.relu, strides=[2, 2], \
            padding='same', **init_params)
        h_re2 = tf.reshape(h_conv2, [-1, 7*7*64])

        d_logit = tf.layers.dense(h_re2, 1, \
            activation=None, **init_params)
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

x_ = tf.placeholder(tf.float32, shape=[None, x_dim])
z_ = tf.placeholder(tf.float32, shape=[None, z_dim])

x_samp = Generator(z_, 'G', False)
d_fake_prob, d_fake_logit = Discriminator(x_samp, 'D', False)
d_real_prob, d_real_logit = Discriminator(x_, 'D', True)
g_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
d_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D')

g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d_fake_logit), logits=d_fake_logit))
d_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_fake_logit), logits=d_fake_logit))
d_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d_real_logit), logits=d_real_logit))
d_loss = d_loss_real + d_loss_fake

g_solver = tf.train.AdamOptimizer(1e-3).minimize(g_loss, var_list=g_var)
d_solver = tf.train.AdamOptimizer(1e-3).minimize(d_loss, var_list=d_var)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

out_folder = 'out_dcgan1/'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
i = 0
batch_size = 64
for it in range(20001):
    x_batch, _ = mnist.train.next_batch(batch_size)
    _, d_loss_ = sess.run([d_solver, d_loss], feed_dict={z_:np.random.randn(batch_size, z_dim), x_:x_batch})
    _, g_loss_ = sess.run([g_solver, g_loss], feed_dict={z_:np.random.randn(batch_size, z_dim)}) 
    
    if it % 100 == 0:
        d_real_prob_, d_fake_prob_ = sess.run([d_real_prob, d_fake_prob], \
        feed_dict={z_:np.random.randn(batch_size, z_dim), x_:x_batch})

        print('Iter: {}'.format(it))
        print('G_loss: {:.4}, D_loss: {:.4}'.format(g_loss_, d_loss_))
        print('D_real: {:.4}, D_fake: {:.4}'.format(np.mean(d_real_prob_), np.mean(d_fake_prob_)))
        print()

        x_samp_ = sess.run(x_samp, feed_dict={z_:np.random.randn(16, z_dim)}) 
        fig = plot(x_samp_, 28, out_folder+str(i).zfill(4))
        i += 1
