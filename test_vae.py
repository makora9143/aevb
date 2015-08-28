#! /usr/bin/env python
# -*- coding: utf-8 -*-


from itertools import cycle
from matplotlib import cm as cm
import matplotlib.pyplot as plt
import numpy as np
import theano

from vae import GaussianVAE, BernoulliVAE
from utils import load_data

def test_vae(
    opt='adagrad',
    n_iters=1000,
    learning_rate=1e-4,
    n_mc_samples=1,
    scale_init=0.01,
    dim_h=100,
    dim_z=2,
    model='Gaussian'):

##################
# load data
##################
    datasets = load_data('../20150717-/mnist.pkl.gz')

    train_set_x, train_set_y = datasets
    xs = train_set_x[:10000]

    sgd_params = {
        'learning_rate' : learning_rate,
        'n_iters'       : n_iters,
        'size_minibatch': 100,
        'calc_hist'     : 'all',
        'n_mod_hist'    : 100,
    }

    adagrad_params = sgd_params

    all_params = {
        'hyper_params': {
            'rng_seed'        : 1234,
            'dim_z'           : dim_z,
            'dim_h_generate'  : dim_h,
            'dim_h_recognize' : dim_h,
            'n_mc_samples'    : n_mc_samples,
            'scale_init'      : scale_init
        }
    }
    if opt == 'adagrad':
        all_params.update({'adagrad_params': adagrad_params})
    elif opt == 'sgd':
        all_params.update({'sgd_params': sgd_params})

    if model == 'Gaussian':
        model = GaussianVAE(**all_params)
    elif model == 'Bernoulli':
        model = BernoulliVAE(**all_params)

    model.fit(xs)

    zs = model.encode(xs)
    xs_recon = model.decode(zs)

    err = np.sum(0.5 * (xs - xs_recon) ** 2) / xs.shape[0]
    print ('Error: %f' % err)

    return datasets, model

def plot_weights(model):
    fig, axes = plt.subplots(nrows=10, ncols=10)
    fig.subplots_adjust(hspace=.001, wspace=.001)
    fig.set_size_inches(10, 10)

    w3 = model.model_params_['w2_'].get_value()
    nx = np.sqrt(w3.shape[1]).astype(int)
    ny = nx
    w3 = w3.reshape((w3.shape[0], ny, nx))

    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(w3[i], interpolation='none', cmap=cm.gray)

def plot_manifold(
    model, z1s=np.arange(-0.8, 1.2, .2), z2s=np.arange(-0.8, 1.2, .2)):
    zs = np.array([[z1, z2] for z2 in z2s
                            for z1 in z1s]).astype(theano.config.floatX)
    xs = model.decode(zs)
    nx = np.sqrt(xs.shape[1]).astype(int)
    ny = nx
    xs = xs.reshape((xs.shape[0], ny, nx))

    fig, axes = plt.subplots(nrows=len(z1s), ncols=len(z2s))
    fig.subplots_adjust(hspace=.001, wspace=.001)
    fig.set_size_inches(10, 10)

    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(xs[i], interpolation='none', cmap=cm.gray)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

def plot_hiddens(model, xs, cs):
    zs = model.encode(xs)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['+', 'o', '^']
    plt.figure(figsize=(7, 7))

    for c, color, marker in zip(np.unique(cs), cycle(colors), cycle(markers)):
        ixs = np.where(cs == c)[0]
        plt.scatter(zs[ixs, 0], zs[ixs, 1], c=color, marker=marker, label=c)

    plt.legend(loc='best', scatterpoints=1, framealpha=1)


if __name__ == '__main__':
    data, model = test_vae(
        n_iters=10000,
        learning_rate=0.01,
        n_mc_samples=1,
        scale_init=1.,
        dim_h=500,
        dim_z=2,
        model='Gaussian',
        opt='adagrad'
    )
    hist = np.vstack(model.hist)
    plt.plot(hist[:, 0], hist[:, 1])

    test_vae.plot_manifold(
        model, z1s=np.arange(-8., 8., 1.), z2s=np.arange(8., -8., -1.))
    plt.show()


# End of Line.

