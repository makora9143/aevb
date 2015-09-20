#! /usr/bin/env python
# -*- coding: utf-8 -*-


from itertools import cycle
from matplotlib import cm as cm
import matplotlib.pyplot as plt
import numpy as np
import theano
from scipy import misc

from utils import load_data
from m1_vae import M1_VAE


def test_vae(
        n_iters=1000,
        learning_rate=1e-4,
        n_mc_samples=1,
        scale_init=0.01,
        dim_z=2,
    ):

    datasets = load_data('../../20150717-/mnist.pkl.gz')

    train_set_x, train_set_y = datasets
    xs = train_set_x[:50000]
    y_index = train_set_y[:50000]
    ys = np.zeros((xs.shape[0], 10)).astype(theano.config.floatX)
    for i in xrange(len(y_index)):
        ys[i][y_index[i]] = 1.

    adagrad_params = {
        'learning_rate' : learning_rate,
        'n_iters'       : n_iters,
        'minibatch_size': 1000,
        'calc_history'     : 'all',
        'calc_hist'     : 'all',
        'n_mod_history'    : 100,
        'n_mod_hist'    : 100,
    }

    all_params = {
        'hyper_params': {
            'rng_seed'          : 1234,
            'dim_z'             : dim_z,
            'n_hidden'          : [500, 500],
            'n_mc_sampling'     : n_mc_samples,
            'scale_init'        : scale_init,
            'nonlinear_q'       : 'tanh',
            'nonlinear_p'       : 'tanh',
            'type_px'           : 'gaussian',
        }
    }
    all_params.update({'adagrad_params': adagrad_params})

    model = M1_VAE(**all_params)
    model.fit(xs)
    zs = model.encode(xs)
    xs_recon = model.decode(zs)

    err = np.sum(0.5 * (xs - xs_recon) ** 2) / xs.shape[0]
    print ('Error: %f' % err)

    return datasets, model

if __name__ == '__main__':
    data, model = test_vae(
        n_iters=10000,
        learning_rate=0.001,
        n_mc_samples=1,
        scale_init=1.,
        dim_z=50,
    )
    hist = np.vstack(model.hist)
    plt.plot(hist[:, 0], hist[:, 1])
    # print model.encode(xs[0], ys[0])

    size = 28
    im_size = (28, 28)
    output_image = np.zeros((size * 10, size * 10))

    for i in range(10):
        for j in range(10):
            sampleZ = np.random.standard_normal((1, 50)).astype(np.float32)
            im = model.decode(sampleZ).reshape(im_size)
            output_image[im_size[0]*i: im_size[0]*(i+1), im_size[1]*j:im_size[1]*(j+1)] = im
    misc.imsave('sampleM1.jpg', output_image)


    plt.show()
# End of Line.
