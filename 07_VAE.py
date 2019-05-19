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

def Encoder(x, name, reuse):
    global x_dim, z_dim
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(x, 256, activation=tf.nn.relu, **init_params)
        z_mu = tf.layers.dense(h1, z_dim, activation=None, **init_params)
        z_logvar = tf.layers.dense(h1, z_dim, activation=None, **init_params)
        return z_mu, z_logvar

def Sample(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def Decoder(z, name, reuse):
    global x_dim, z_dim
    with tf.variable_scope(name, reuse=reuse):
        h1 = tf.layers.dense(z, 256, activation=tf.nn.relu, **init_params)
        h2 = tf.layers.dense(h1, x_dim, activation=None, **init_params)
        x_out = tf.nn.sigmoid(h2)
        return x_out

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

z_mu, z_logvar = Encoder(x_, 'Enc', False)
z_samp = Sample(z_mu, z_logvar)
x_re = Decoder(z_samp, 'Dec', False)
x_samp = Decoder(z_, 'Dec', True)

Rec_loss = tf.reduce_mean(tf.square(x_ - x_re))
KL_loss = 0.5 * tf.reduce_mean(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
VAE_loss = Rec_loss + 0.2*KL_loss

solver = tf.train.AdamOptimizer().minimize(VAE_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

out_folder = 'out_vae/'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
i = 0
batch_size = 64
for it in range(20001):
    x_batch, _ = mnist.train.next_batch(batch_size)
    _, loss_vae, loss_kl, loss_rec = sess.run([solver, VAE_loss, KL_loss, Rec_loss], feed_dict={x_: x_batch})

    if it % 100 == 0:
        print('Iter: {}, Loss: {:.4}, KL_Loss: {:.4}, Rec_Loss: {:.4}'.format(it, loss_vae, loss_kl, loss_rec))
        
        reconst = sess.run(x_re, feed_dict={x_: x_batch[:4]})
        samples = sess.run(x_samp, feed_dict={z_: np.random.randn(8, z_dim)})
        out_img = np.concatenate([x_batch[:4], reconst, samples], 0)
        fig = plot(out_img, 28, out_folder+str(i).zfill(4))
        i += 1
