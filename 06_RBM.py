import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

v_dim = 784
h_dim = 64

w = np.random.randn(v_dim, h_dim) / np.sqrt(v_dim)
b = np.ones(h_dim) * 1e-3
c = np.ones(v_dim) * 1e-3

def Sigmoid(x):
    return 1. / (1. + np.exp(-x))

def Encode(v):
    global w,b,c
    return Sigmoid(v.dot(w) + b)

def Decode(h):
    global w,b,c
    return Sigmoid(h.dot(w.T) + c)

def GibbsSample(iter):
    global w,b,c
    h_samp = np.random.choice(2,h_dim).astype(np.float32)
    v_exp = Decode(h_samp)
    v_samp = np.random.binomial(n=1, p=v_exp) # v~p(v|h)
    
    for i in range(iter):
        h_exp = Encode(v_samp)
        h_samp = np.random.binomial(n=1, p=h_exp) # h~p(h|v)
        v_exp = Decode(h_samp)
        v_samp = np.random.binomial(n=1, p=v_exp) # v~p(v|h)
    return v_samp

def CD_k(v_batch, k, lr=0.1):
    global w,b,c
    grad_w, grad_b, grad_c = 0, 0, 0

    for v in v_batch:
        h_exp_v0 = Encode(v) # E[h|v=v0]

        # k-step MCMC (Gibbs) Sampling 
        v_samp = np.copy(v)
        for i in range(k):
            h_exp = Encode(v_samp)
            h_samp = np.random.binomial(n=1, p=h_exp) # h~p(h|v)
            v_exp = Decode(h_samp)
            v_samp = np.random.binomial(n=1, p=v_exp) # v~p(v|h)
        h_exp_vk = Encode(v_samp) # E[h|v=vk]

        # Calculate Gradient <v0,h0> - <vk,hk>
        grad_w += np.outer(v, h_exp_v0) - np.outer(v_samp, h_exp_vk)
        grad_b += h_exp_v0 - h_exp_vk
        grad_c += v - v_exp
    
    # Update
    batch_size = v_batch.shape[0]
    w += lr * grad_w / batch_size
    b += lr * grad_b / batch_size
    c += lr * grad_c / batch_size

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

    plt.savefig(out_file+'{}.png'.format(name), bbox_inches='tight')
    plt.close(fig)

# main
if __name__ == '__main__':
    out_file = 'out_rbm/'
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    batch_size = 64
    for i in range(20001):
        v_batch = (mnist.train.next_batch(batch_size)[0] > 0.5).astype(np.float)
        CD_k(v_batch, k=1, lr=1e-2)
    
        if i % 100 == 0:
            print(i)
            id = str(int(i/100)).zfill(4)
            v_batch = (mnist.train.next_batch(16)[0] > 0.5).astype(np.float)
            h_batch = np.random.binomial(n=1, p=Encode(v_batch))
            v_recon = (Decode(h_batch) > 0.5).astype(np.float)

            plot(v_recon, np.sqrt(v_dim), id+'_1_V_r')
            plot(v_batch, np.sqrt(v_dim), id+'_2_V')
            plot(h_batch, np.sqrt(h_dim), id+'_3_H')
            plot(w.T, np.sqrt(v_dim), id+'_4_W', figsize=(8,8))
            
        '''
        if i % 1000 == 0:
            # Gibbs Sampling
            v_list = []
            for i in range(16):
                v_list.append(GibbsSample(50000))
            plot(np.asarray(v_list), np.sqrt(v_dim), 'Sample_'+id)
        '''
    
