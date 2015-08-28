#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gzip
import cPickle

def load_data(dataset='mnist.pkl.gz'):
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        print 'Downloading mnist data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set


# End of Line.
