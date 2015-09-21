#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import Layer
from vae import Base_VAE


class M2_VAE(Base_VAE):

    def init_model_params(self, dim_x, dim_y):
        print 'M2 model params initialize'

        dim_z = self.hyper_params['dim_z']
        n_hidden = self.hyper_params['n_hidden'] # [500, 500, 500]
        n_hidden_recognize = n_hidden
        n_hidden_generate = n_hidden[::-1]

        self.type_px = self.hyper_params['type_px']

        activation = {
            'tanh': T.tanh,
            'relu': self.relu,
            'softplus': self.softplus,
            'sigmoid': T.nnet.sigmoid,
            'none': self.identify,
        }

        self.nonlinear_q = activation[self.hyper_params['nonlinear_q']]
        self.nonlinear_p = activation[self.hyper_params['nonlinear_p']]
        if self.type_px == 'bernoulli':
            output_f = activation['sigmoid']
        elif self.type_px == 'gaussian':
            output_f= activation['none']

        # Recognize model
        self.recognize_layers = [
            Layer(param_shape=(dim_x, n_hidden_recognize[0]), function=self.identify, nonbias=True),
            Layer(param_shape=(dim_y, n_hidden_recognize[0]), function=self.identify)
        ]
        if len(n_hidden_recognize) > 1:
            self.recognize_layers += [
                Layer(param_shape=shape, function=self.nonlinear_q)
                for shape in zip(n_hidden_recognize[:-1], n_hidden_recognize[1:])
            ]
        self.recognize_mean_layer = Layer(
            param_shape=(n_hidden_recognize[-1], dim_z),
            function=self.identify
        )
        self.recognize_log_var_layer = Layer(
            param_shape=(n_hidden_recognize[-1], dim_z),
            function=self.identify,
            w_zero=True, b_zero=True
        )

        # Generate Model
        self.generate_layers = [
            Layer((dim_z, n_hidden_generate[0]), function=self.identify, nonbias=True),
            Layer((dim_y, n_hidden_generate[0]), function=self.identify),
        ]
        if len(n_hidden) > 1:
            self.generate_layers += [
                Layer(param_shape=shape, function=self.nonlinear_p)
                for shape in zip(n_hidden_generate[:-1], n_hidden_generate[1:])
            ]
        self.generate_mean_layer = Layer(
            param_shape=(n_hidden_generate[-1], dim_x),
            function=output_f
        )
        self.generate_log_var_layer = Layer(
            param_shape=(n_hidden_generate[-1], dim_x),
            function=self.identify,
            b_zero=True
        )

        # Add all parameters
        self.model_params_ = (
            [param for layer in self.recognize_layers for param in layer.params] +
            self.recognize_mean_layer.params +
            self.recognize_log_var_layer.params +
            [param for layer in self.generate_layers for param in layer.params] +
            self.generate_mean_layer.params
        )

        if self.type_px == 'gaussian':
            self.model_params_ += self.generate_log_var_layer.params

    def recognize_model(self, X, Y):
        for i, layer in enumerate(self.recognize_layers):
            if i == 0:
                layer_out = layer.fprop(X)
            elif i == 1:
                layer_out += layer.fprop(Y)
                layer_out = self.nonlinear_q(layer_out)
            else:
                layer_out = layer.fprop(layer_out)

        q_mean = self.recognize_mean_layer.fprop(layer_out)
        q_log_var = self.recognize_log_var_layer.fprop(layer_out)

        return {
            'q_mean': q_mean,
            'q_log_var': q_log_var,
        }

    def generate_model(self, Z, Y):
        for i, layer in enumerate(self.generate_layers):
            if i == 0:
                layer_out = layer.fprop(Z)
            elif i == 1:
                layer_out += layer.fprop(Y)
                layer_out = self.nonlinear_p(layer_out)
            else:
                layer_out = layer.fprop(layer_out)

        p_mean = self.generate_mean_layer.fprop(layer_out)
        p_log_var = self.generate_log_var_layer.fprop(layer_out)

        return {
            'p_mean': p_mean,
            'p_log_var': p_log_var
        }

    def encode(self, x, y):
        if self.encode_main is None:
            X = T.matrix()
            Y = T.matrix()
            self.encode_main = theano.function(
                inputs=[X, Y],
                outputs=self.recognize_model(X, Y)['q_mean']
            )
        return self.encode_main(x, y)

    def decode(self, z, y):
        if self.decode_main is None:
            Z = T.matrix()
            Y = T.matrix()
            self.decode_main = theano.function(
                inputs=[Z, Y],
                outputs=self.generate_model(Z, Y)['p_mean']
            )
        return self.decode_main(z, y)

    def get_expr_lbound(self, X, Y):
        n_samples = X.shape[0]

        recognized_zs = self.recognize_model(X, Y)
        q_mean = recognized_zs['q_mean']
        q_log_var = recognized_zs['q_log_var']

        eps = self.rng_noise.normal(avg=0., std=1., size=q_mean.shape).astype(theano.config.floatX)
        # T.exp(0.5 * q_log_var) = std
        # z = mean_z + std * epsilon
        z_tilda = q_mean + T.exp(0.5 * q_log_var) * eps

        generated_x = self.generate_model(z_tilda, Y)
        p_mean = generated_x['p_mean']
        p_log_var = generated_x['p_log_var']

        if self.type_px == 'gaussian':
            log_p_x_given_z = (
                - 0.5 * np.log(2 * np.pi) -
                0.5 * p_log_var -
                0.5 * (X - p_mean) ** 2 / (2 * T.exp(p_log_var))
            )
        elif self.type_px == 'bernoulli':
            # log_p_x_given_z = X * T.log(p_mean) + (1 - X) * T.log(1 - p_mean)
            log_p_x_given_z = - T.nnet.binary_crossentropy(p_mean, X)

        logqz = - 0.5 * (np.log(2 * np.pi) + 1 + q_log_var)
        logpz = - 0.5 * (np.log(2 * np.pi) + q_mean ** 2 + T.exp(q_log_var))
        # logqz = - 0.5 * T.sum(np.log(2 * np.pi) + 1 + q_log_var, axis=1)
        # logpz = - 0.5 * T.sum(np.log(2 * np.pi) + q_mean ** 2 + T.exp(q_log_var), axis=1)
        D_KL = T.sum(logpz - logqz)
        recon_error = T.sum(log_p_x_given_z)

        return D_KL, recon_error
        # return log_p_x_given_z, logpz, logqz

    def fit(self, x_datas, y_labels):
        X = T.matrix()
        Y = T.matrix()
        self.rng_noise = RandomStreams(self.hyper_params['rng_seed'])
        self.init_model_params(dim_x=x_datas.shape[1], dim_y=y_labels.shape[1])

        D_KL, recon_error = self.get_expr_lbound(X, Y)
        L = D_KL + recon_error

        print 'start fitting'
        gparams = T.grad(
            cost=L,
            wrt=self.model_params_
        )

        optimizer = {
            'sgd': self.sgd,
            'adagrad': self.adagrad,
            'adadelta': self.adaDelta,
            'rmsprop': self.rmsProp,
            'adam': self.adam
        }

        updates = optimizer[self.hyper_params['optimizer']](
            self.model_params_, gparams, self.optimize_params)
        self.hist = self.early_stopping(
        # self.hist = self.optimize(
            X,
            Y,
            x_datas,
            y_labels,
            self.optimize_params,
            L,
            updates,
            self.rng,
            D_KL,
            recon_error,
        )

    def optimize(self, X, Y, x_datas, y_labels, hyper_params, cost, updates, rng, D_KL, recon_error):
        n_iters = hyper_params['n_iters']
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']

        train_x = x_datas[:50000]
        valid_x = x_datas[50000:]

        train_y = y_labels[:50000]
        valid_y = y_labels[50000:]

        train = theano.function(
            inputs=[X, Y],
            outputs=[cost, D_KL, recon_error],
            updates=updates
        )

        validate = theano.function(
            inputs=[X, Y],
            outputs=[cost, D_KL, recon_error]
        )

        n_samples = train_x.shape[0]
        cost_history = []

        total_cost = 0
        total_dkl = 0
        total_recon_error = 0
        for i in xrange(n_iters):
            ixs = rng.permutation(n_samples)
            for j in xrange(0, n_samples, minibatch_size):
                cost, D_KL, recon_error = train(train_x[ixs[j:j+minibatch_size]], train_y[ixs[j:j+minibatch_size]])
                # print np.sum(hoge(train_x[:1])[0])
                total_cost += cost
                total_dkl += D_KL
                total_recon_error += recon_error

            if np.mod(i, n_mod_history) == 0:
                num = n_samples / minibatch_size
                print ('%d epoch train D_KL error: %.3f, Reconstruction error: %.3f, total error: %.3f' %
                      (i, total_dkl / num, total_recon_error / num, total_cost / num))
                total_cost = 0
                total_dkl = 0
                total_recon_error = 0
                valid_error, valid_dkl, valid_recon_error = validate(valid_x, valid_y)
                print '\tvalid D_KL error: %.3f, Reconstruction error: %.3f, total error: %.3f' % (valid_dkl, valid_recon_error, valid_error)
                cost_history.append((i, valid_error))
        return cost_history

    def early_stopping(self, X, Y, x_datas, y_labels, hyper_params, cost, updates, rng, D_KL, recon_error):
        minibatch_size = hyper_params['minibatch_size']
        n_mod_history = hyper_params['n_mod_history']
        calc_history = hyper_params['calc_history']

        train_x = x_datas[:50000]
        valid_x = x_datas[50000:]

        train_y = y_labels[:50000]
        valid_y = y_labels[50000:]

        train = theano.function(
            inputs=[X, Y],
            outputs=[cost, D_KL, recon_error],
            updates=updates
        )

        validate = theano.function(
            inputs=[X, Y],
            outputs=cost,
        )

        n_samples = train_x.shape[0]
        cost_history = []
        best_params = None
        valid_best_error = - np.inf
        best_iter = 0
        patience = 5000
        patience_increase = 2
        improvement_threshold = 1.005

        done_looping = False

        for i in xrange(1000000):
            if done_looping: break
            ixs = rng.permutation(n_samples)
            for j in xrange(0, n_samples, minibatch_size):
                cost, D_KL, recon_error = train(train_x[ixs[j:j+minibatch_size]], train_y[ixs[j:j+minibatch_size]])

                iter = i * (n_samples / minibatch_size) + minibatch_size

                if (iter + 1) % 50 == 0:
                    valid_error = 0.
                    for _ in xrange(3):
                        valid_error += validate(valid_x, valid_y)
                    valid_error /= 3
                    print 'epoch %d, minibatch %d/%d, valid total error: %.3f' % (i, j / minibatch_size + 1, n_samples / minibatch_size, valid_error)
                    cost_history.append((i, valid_error))
                    if valid_error > valid_best_error:
                        if valid_error > valid_best_error * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_params = self.model_params_
                        valid_best_error = valid_error

                if patience <= iter:
                    done_looping = True
                    break
        return cost_history



# End of Line.
