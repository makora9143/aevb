#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class VariationalAutoencoder(object):
    def __init__(
                 self,
                 hyper_params=None,
                 sgd_params=None,
                 adagrad_params=None,
                 model_params=None):

        if (sgd_params is not None) and (adagrad_params is not None):
            raise ValueError("""Error: select only one algorithm.""")

        self.hyper_params = hyper_params
        self.sgd_params = sgd_params
        self.adagrad_params = adagrad_params
        self.model_params = model_params

        self.rng = np.random.RandomState(hyper_params['rng_seed'])

        self.model_params_ = None
        self._decode_main = None
        self._encode_main = None

    def fit(self, xs):
        xs_ = T.matrix()
        self.rng_th = RandomStreams(self.hyper_params['rng_seed'])
        self._init_model_params(dim_x=xs.shape[1])

        lbound, consts = self._get_expr_lbound(xs_)
        cost = -lbound
        hoge = theano.function(
            inputs=[xs_],
            outputs=[lbound]
        )

        print hoge(xs)
        cost = -lbound

        model_params_list = [p for p in self.model_params_.values()]

        print T.grad(cost, model_params_list)

        if self.sgd_params is not None:
            self.hist = sgd_xs(
                xs,
                cost,
                consts,
                xs_,
                model_params_list,
                self.sgd_params,
                self.rng
            )

        else:
            self.hist = adagrad_xs(
                xs,
                cost,
                consts,
                xs_,
                model_params_list,
                self.adagrad_params,
                self.rng,
            )


class GaussianVAE(VariationalAutoencoder):
    """
    An example of :code: 'hyper_params':

    .. code:: python

        hyper_params = {
            'rng_seed'        : 1234,
            'dim_z'           : 2,
            'dim_h_generate'  : 500,
            'dim_h_recognize' : 500,
            'n_mc_samples'    : 1
        }

    An example of :code: 'model_params':

    .. code:: python

        dim_z = self.hyper_params['dim_z']
        dim_h_generate = self.hyper_params['dim_h_generate']
        dim_h_recognize = self.hyper_params['dim_h_recognize']
        rng = np.random.RandomState(self.hyper_params['rng_seed'])

        N = lambda size: rng.normal(
            scale=0.01, size=size).astype(theano.config.theano)

        model_params = {
            'w1': N((dim_z, dim_h_generate)),
            'w2': N((dim_h_generate, dim_x)),
            'w3': N((dim_h_generate, dim_x)),
            'b1': N((dim_h_generate,))
            'b2': N((dim_x,))
            'b3': N((dim_x,))

            'w4': N((dim_x, dim_h_recognize)),
            'w5': N((dim_h_recognize, dim_z)),
            'w6': N((dim_h_recognize, dim_z)),
            'b4': N((dim_h_recognize,))
            'b5': N((dim_z,))
            'b6': N((dim_z,))
        }

    An example of :code: 'sgd_params':

    .. code:: python
        sgd_params = {
            'n_iters'        : 1000,
            'learning_rate'  : 0.01,
            'size_minibatch' : 100,
            'n_mod_hist'     : 10,
            'calc_hist'      : 'all'
        }
    """
    # def __init__(
    #     self,
    #     hyper_params=None,
    #     sgd_params=None,
    #     adagrad_params=None,
    #     model_params=None
    #     ):

    #     VariationalAutoencoder.__init__(
    #         self,
    #         hyper_params,
    #         sgd_params,
    #         adagrad_params,
    #         model_params
    #     )

    def _init_model_params(self, dim_x):
        if self.model_params is not None:
            print 'model_params is not None'
            w1 = self.model_params['w1'].astype(theano.config.floatX)
            w2 = self.model_params['w2'].astype(theano.config.floatX)
            w3 = self.model_params['w3'].astype(theano.config.floatX)
            b1 = self.model_params['b1'].astype(theano.config.floatX)
            b2 = self.model_params['b2'].astype(theano.config.floatX)
            b3 = self.model_params['b3'].astype(theano.config.floatX)

            w4 = self.model_params['w4'].astype(theano.config.floatX)
            w5 = self.model_params['w5'].astype(theano.config.floatX)
            w6 = self.model_params['w6'].astype(theano.config.floatX)
            b4 = self.model_params['b4'].astype(theano.config.floatX)
            b5 = self.model_params['b5'].astype(theano.config.floatX)
            b6 = self.model_params['b6'].astype(theano.config.floatX)
        else:
            print 'model_params is None'
            dim_z = self.hyper_params['dim_z']
            dim_h_generate = self.hyper_params['dim_h_generate']
            dim_h_recognize = self.hyper_params['dim_h_recognize']

            N = lambda size: self.rng.normal(
                scale=self.hyper_params['scale_init'],
                size=size).astype(theano.config.floatX)
            U = lambda size: self.rng.uniform(
                low=-np.sqrt(6. / np.sum(size)),
                high=np.sqrt(6. / np.sum(size)),
                size=size).astype(theano.config.floatX)

            w1 = U((dim_z, dim_h_generate))
            w2 = U((dim_h_generate, dim_x))
            w3 = U((dim_h_generate, dim_x))
            b1 = N((dim_h_generate,))
            b2 = N((dim_x,))
            b3 = N((dim_x,))

            w4 = U((dim_x, dim_h_recognize))
            w5 = U((dim_h_recognize, dim_z))
            w6 = U((dim_h_recognize, dim_z))
            b4 = N((dim_h_recognize,))
            b5 = N((dim_z,))
            b6 = N((dim_z,))

        self.model_params_ = {
            'w1': theano.shared(w1),
            'w2': theano.shared(w2),
            'w3': theano.shared(w3),
            'b1': theano.shared(b1),
            'b2': theano.shared(b2),
            'b3': theano.shared(b3),

            'w4': theano.shared(w4),
            'w5': theano.shared(w5),
            'w6': theano.shared(w6),
            'b4': theano.shared(b4),
            'b5': theano.shared(b5),
            'b6': theano.shared(b6),
        }

    def _generate_model(self, zs_):
        w1 = self.model_params_['w1']
        w2 = self.model_params_['w2']
        w3 = self.model_params_['w3']
        b1 = self.model_params_['b1']
        b2 = self.model_params_['b2']
        b3 = self.model_params_['b3']

        hs = T.tanh(T.dot(zs_, w1) + b1)

        return {
            'mu': 0.5 * (T.tanh(T.dot(hs, w2) + b2) + 1),
            'log_sigma2': 3 * T.tanh(T.dot(hs, w3) + b3) - 1
        }

    def _recognize_model(self, xs_):
        w4 = self.model_params_['w4']
        w5 = self.model_params_['w5']
        w6 = self.model_params_['w6']
        b4 = self.model_params_['b4']
        b5 = self.model_params_['b5']
        b6 = self.model_params_['b6']

        hs = T.tanh(T.dot(xs_, w4) + b4)

        return {
            'mu': T.dot(hs, w5) + b5,
            'log_sigma2': 3 * T.tanh(T.dot(hs, w6) + b6) - 1
        }

    def _get_expr_lbound(self, xs_):
        n_mc_samples = self.hyper_params['n_mc_samples']
        n_samples = xs_.shape[0]
        dim_z = self.hyper_params['dim_z']

        stats_z = self._recognize_model(xs_)
        mu_z = stats_z['mu']
        log_sigma2_z = stats_z['log_sigma2']
        sigma2_z = T.exp(log_sigma2_z)

        # epsilon
        eps = self.rng_th.normal(size=(n_mc_samples, n_samples, dim_z))
        zss = mu_z + T.sqrt(sigma2_z) * eps

        stats_x = self._generate_model(zss)
        mus_x = stats_x['mu']
        log_sigma2_x = stats_x['log_sigma2']
        sigmas2_x = T.exp(log_sigma2_x)
        # log p(x|z) = log N(x; mu, sigma**2)
        logp_x_given_z = (
            - 0.5 * T.log(sigmas2_x) - 0.5 * (xs_ - mus_x)**2 / sigmas2_x
        )

        consts = []
        # equation (10)
        return (
            0.5 * T.sum(1 + log_sigma2_z - mu_z**2 - sigma2_z) / n_samples + # KL Divergense
            T.sum(logp_x_given_z) / (n_mc_samples * n_samples)
        ), consts

    def decode(self, zs):
        if self._decode_main is None:
            zs_ = T.matrix()
            self._decode_main = theano.function(
                inputs=[zs_],
                outputs=self._generate_model(zs_)['mu'])
        return self._decode_main(zs)

    def encode(self, xs):
        if self._encode_main is None:
            if self.model_params_ is None:
                self._init_model_params()
            xs_ = T.matrix()
            self._encode_main = theano.function(
                inputs=[xs_],
                outputs=self._recognize_model(xs_)['mu'])
        return self._encode_main(xs)

class BernoulliVAE(VariationalAutoencoder):
    def __init__(
        self,
        hyper_params=None,
        sgd_params=None,
        adagrad_params=None,
        model_params=None
        ):

        VariationalAutoencoder.__init__(
            self,
            hyper_params,
            sgd_params,
            adagrad_params,
            model_params
        )

    def _init_model_params(self, dim_x):
        if self.model_params is not None:
            w1 = self.model_params['w1'].astype(theano.config.floatX)
            w2 = self.model_params['w2'].astype(theano.config.floatX)
            b1 = self.model_params['b1'].astype(theano.config.floatX)
            b2 = self.model_params['b2'].astype(theano.config.floatX)

            w4 = self.model_params['w4'].astype(theano.config.floatX)
            w5 = self.model_params['w5'].astype(theano.config.floatX)
            w6 = self.model_params['w6'].astype(theano.config.floatX)
            b4 = self.model_params['b4'].astype(theano.config.floatX)
            b5 = self.model_params['b5'].astype(theano.config.floatX)
            b6 = self.model_params['b6'].astype(theano.config.floatX)
        else:
            dim_z = self.hyper_params['dim_z']
            dim_h_generate = self.hyper_params['dim_h_generate']
            dim_h_recognize = self.hyper_params['dim_h_recognize']

            N = lambda size: self.rng.normal(
                scale=self.hyper_params['scale_init'],
                size=size).astype(theano.config.floatX)
            U = lambda size: self.rng.uniform(
                low=-np.sqrt(6. / np.sum(size)),
                high=np.sqrt(6. / np.sum(size)),
                size=size).astype(theano.config.floatX)

            w1 = U((dim_z, dim_h_generate))
            w2 = U((dim_h_generate, dim_x))
            b1 = N((dim_h_generate,))
            b2 = N((dim_x,))

            w4 = U((dim_x, dim_h_recognize))
            w5 = U((dim_h_recognize, dim_z))
            w6 = U((dim_h_recognize, dim_z))
            b4 = N((dim_h_recognize,))
            b5 = N((dim_z,))
            b6 = N((dim_z,))

        self.model_param_ = {
            'w1': theano.shared(w1),
            'w2': theano.shared(w2),
            'b1': theano.shared(b1),
            'b2': theano.shared(b2),

            'w4': theano.shared(w4),
            'w5': theano.shared(w5),
            'w6': theano.shared(w6),
            'b4': theano.shared(b4),
            'b5': theano.shared(b5),
            'b6': theano.shared(b6),
        }

    def _generate_model(self, zs_):
        w1 = self.model_param_['w1']
        w2 = self.model_param_['w2']
        b1 = self.model_param_['b1']
        b2 = self.model_param_['b2']

        hs = T.tanh(T.dot(zs_, w1) + b1)

        return {
            'y': T.nnet.sigmoid(T.dot(hs, w2) + b2)
        }

    def _recognize_model(self, xs_):
        w4 = self.model_param_['w4']
        w5 = self.model_param_['w5']
        w6 = self.model_param_['w6']
        b4 = self.model_param_['b4']
        b5 = self.model_param_['b5']
        b6 = self.model_param_['b6']

        hs = T.tanh(T.dot(xs_, w4) + b4)

        return {
            'mu': T.dot(hs, w5) + b5,
            'log_sigma2': 3 * T.tanh(T.dot(hs, w6) + b6) - 1
        }

    def _get_expr_lbound(self, xs_):
        n_mc_samples = self.hyper_params['n_mc_samples']
        n_samples = xs_.shape[0]
        dim_z = self.hyper_params['dim_z']

        stats_z = self._recognize_model(xs_)
        mu_z = stats_z['mu']
        log_sigma2_z = stats_z['log_sigma2']
        sigma2_z = T.exp(log_sigma2_z)
        sigma_z = T.sqrt(sigma2_z)

        eps = self.rng_th.normal(size=(n_mc_samples, n_samples, dim_z))
        zss = mu_z + sigma_z * eps

        stats_x = self._generate_model(zss)
        ys = stats_x['y']
        logp_x_given_z_ = xs_ * T.log(ys) + (1 - xs_) * T.log(1 - ys)

        consts = []

        return (
            0.5 * T.sum(1 + log_sigma2_z - mu_z**2 - sigma2_z) / n_samples +
            T.sum(logp_x_given_z_) / (n_mc_samples * n_samples)
            ), consts

    def decode(self, zs):
        if self._decode_main is None:
            self._init_model_params()
            zs_ = T.matrix()
            self._decode_main = theano.function(
                inputs=[zs_],
                outputs=self._generate_model(zs_)['y']
            )
            return self._decode_main(zs)

    def encode(self, xs):
        if self._encode_main is None:
            if self.model_params_ is None:
                self._init_model_params()
            xs_ = T.matrix()
            self._encode_main = theano.function(
                inputs=[xs_],
                outputs=self._recognize_model(xs_)['mu'])
        return self._encode_main(xs)


def sgd_xs(xs, cost, consts, xs_, model_params_, sgd_params, rng):
    """
    An example of :code: 'sgd_params':

    .. code:: python
        sgd_params = {
            'n_iters'        : 1000,
            'learning_rate'  : 0.01,
            'size_minibatch' : 100,
            'n_mod_hist'     : 10,
            'calc_hist'      : 'all'
        }
    """
    n_iters = sgd_params['n_iters']
    learning_rate = sgd_params['learning_rate']
    size_minibatch = sgd_params['size_minibatch']
    n_mod_hist = sgd_params['n_mod_hist']
    calc_hist = sgd_params['calc_hist']

    grads_ = T.grad(cost=cost, wrt=model_params_, consider_constant=consts)
    updates = [(p, p - learning_rate * g) for p, g in zip(model_params_, grads_)]
    pyfnc_update = theano.function(
        inputs=[xs_],
        outputs=cost,
        updates=updates
    )
    pyfunc_cost = theano.function(
        inputs=[xs_],
        outputs=cost
    )

    n_samples = xs.shape[0]
    hist = []
    for i in xrange(n_iters):
        ixs = rng.permutation(n_samples)[:size_minibatch]
        cost_minibatch = pyfnc_update(xs[ixs])

        if np.mod(i, n_mod_hist) == 0:
            print '%d epoch' % i
            if calc_hist == 'minibatch':
                hist.append((i, cost_minibatch))
            elif calc_hist == 'all':
                hist.append((i, pyfunc_cost(xs)))
    return hist

def adagrad_xs(xs, cost_, consts_, xs_, model_params_, adagrad_params, rng):
    n_iters = adagrad_params['n_iters']
    learning_rate = adagrad_params['learning_rate']
    size_minibatch = adagrad_params['size_minibatch']
    n_mod_hist = adagrad_params['n_mod_hist']
    calc_hist = adagrad_params['calc_hist']

    hs_ = [theano.shared(
            np.ones(
                p.get_value(borrow=True).shape
            ).astype(theano.config.floatX)
          ) for p in model_params_]

    grads_ = T.grad(cost=cost_, wrt=model_params_, consider_constant=consts_)
    updates = ([(p, p - learning_rate / T.sqrt(h) * g)
                for p, g, h in zip(model_params_, grads_, hs_)] +
               [(h, h + g**2) for g, h in zip(grads_, hs_)])

    pyfnc_update = theano.function(
        inputs=[xs_],
        outputs=cost_,
        updates=updates
    )
    pyfunc_cost = theano.function(
        inputs=[xs_],
        outputs=cost_
    )

    n_samples = xs.shape[0]
    hist = []
    for i in xrange(n_iters):
        ixs = rng.permutation(n_samples)[:size_minibatch]
        cost_minibatch = pyfnc_update(xs[ixs])

        if np.mod(i, n_mod_hist) == 0:
            print '%d epoch' % i
            if calc_hist == 'minibatch':
                hist.append((i, cost_minibatch))
            elif calc_hist == 'all':
                hist.append((i, pyfunc_cost(xs)))
    return hist

# End of Line.
