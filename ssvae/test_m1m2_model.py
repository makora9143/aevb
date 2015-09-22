#! /usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import theano
from scipy import misc

from utils import load_data
from m1_vae import M1_VAE
from m2_vae import M2_VAE

datasets = load_data('../../20150717-/mnist.pkl.gz')
train_set, validate_set = datasets
train_x, train_y = train_set
validate_x, validate_y = validate_set
xs = np.r_[train_x, validate_x]
y_index = np.r_[train_y, validate_y]
ys = np.zeros((xs.shape[0], 10)).astype(theano.config.floatX)
for i in xrange(len(y_index)):
    ys[i][y_index[i]] = 1.

def test_vae(
        n_iters=1000,
        learning_rate=1e-4,
        n_mc_samples=1,
        scale_init=0.01,
        dim_z=2,
    ):

    optimize_params = {
        'learning_rate' : learning_rate,
        'n_iters'       : n_iters,
        'minibatch_size': 1000,
        'calc_history'     : 'all',
        'calc_hist'     : 'all',
        'n_mod_history'    : 100,
        'n_mod_hist'    : 100,
        'patience'      : 5000,
        'patience_increase': 2,
        'improvement_threshold': 1.005,
    }

    all_params = {
        'hyper_params': {
            'rng_seed'          : 1234,
            'dim_z'             : dim_z,
            'n_hidden'          : [500, 500],
            'n_mc_sampling'     : n_mc_samples,
            'scale_init'        : scale_init,
            'nonlinear_q'       : 'softplus',
            'nonlinear_p'       : 'softplus',
            'type_px'           : 'bernoulli',
            'optimizer'         : 'adam',
            'learning_process'  : 'early_stopping'
        }
    }
    all_params.update({'optimize_params': optimize_params})

    m1model = M1_VAE(**all_params)
    m2model = M2_VAE(**all_params)


    m1model.fit(xs)
    z1s = m1model.encode(xs)

    m2model.fit(z1s, ys)
    # z2s = m2model.encode(z1s, ys)
    # z1_recon = m2model.decode(z2s, ys)

    # xs_recon = m1model.decode(z1_recon)

    # err = np.sum(0.5 * (xs - xs_recon) ** 2) / xs.shape[0]
    # print ('Error: %f' % err)

    return datasets, m1model, m2model

if __name__ == '__main__':
    data, m1model, m2model = test_vae(
        n_iters=1000,
        learning_rate=0.001,
        n_mc_samples=1,
        scale_init=1.,
        dim_z=50,
    )
    m1hist = np.vstack(m1model.hist)
    plt.plot(m1hist[:, 0], m1hist[:, 1], 'b')
    m2hist = np.vstack(m2model.hist)
    plt.plot(m2hist[:, 0], m2hist[:, 1], 'r')

    size = 28
    im_size = (28, 28)
    output_image = np.zeros((size * 10, size * 11))
    for i in range(10):
        idx = np.random.randint(xs.shape[0])
        testX = [xs[idx]]
        output_image[im_size[0]*i: im_size[0]*(i+1), im_size[1]*(0):im_size[1]*(1)] = np.array(testX).reshape(im_size)
        testY = [ys[idx]]
        testZ1 = m1model.encode(testX)
        testZ2 = m2model.encode(testZ1, testY)
        for j in range(ys.shape[1]):
            sampleY = np.zeros((1, ys.shape[1])).astype(np.float32)
            sampleY[0][j] = 1.
            testZ1 = m2model.decode(testZ2, sampleY)
            im = m1model.decode(testZ1).reshape(im_size)
            output_image[im_size[0]*i: im_size[0]*(i+1), im_size[1]*(j+1):im_size[1]*(j+2)] = im
    misc.imsave('sample' + m1model.get_name() + m2model.get_name() + '.jpg', output_image)

    plt.show()
# End of Line.
