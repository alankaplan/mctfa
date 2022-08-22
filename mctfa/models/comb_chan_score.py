#!/usr/bin/env python

import h5py
import numpy as np
import datetime
from mctfa.utils import sigUtils


def main(data_file, group, f, mu, s, t_int=2.0):
    #
    #   model A: combined model
    #
    #

    spec_options = {'norm': True, 'pruneZ': False, 'remDC': True,
                    'nperseg': t_int, 'noverlap': float(t_int)/2.}
    notch = [(60, 4), (120, 4), (180, 4), (240, 4), (80, 4),
             (200, 4), (220, 4)]

    test_dat = h5py.File(data_file, 'r')

    (x, f) = sigUtils.compute_spectra_mult_fs(test_dat[group], 500,
                                              notch=notch,
                                              spec_options=spec_options)
    (E, N, S, F) = x.shape
    x_min = min([i for i in x.flatten() if i > 0])
    x = np.log(x + 0.01*x_min)

    # Score model
    mu_x = np.dot(np.ones((E, N, S, 1)), mu[None, :])
    s_x = np.dot(np.ones((E, N, S, 1)), s[None, :])
    y0 = -0.5*np.log(2*np.pi) - 0.5*np.log(s_x)
    y1 = -0.5*((x - mu_x)**2)/s_x
    scores_indiv = y0 + y1
    scores = np.nansum(scores_indiv, axis=(1, 2, 3))/(N*S*F)

    test_dat.close()

    return (scores_indiv, scores)


if __name__ == '__main__':
    import argparse
    from mctfa.data.results import dat

    time_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    parser = argparse.ArgumentParser(
                description='Combined channel model scoring')
    parser.add_argument(action='store', dest='T_PATH',
                        help='Path to data file')
    parser.add_argument(action='store', dest='GROUP',
                        help='HDF5 group in the data file to be used')
    parser.add_argument(action='store', dest='MODEL',
                        help='Path to saved model')
    parser.add_argument(action='store', dest='S_PATH',
                        help='Path to save file')
    results = parser.parse_args()

    x = dat()
    x.load(results.MODEL)
    f = x.f
    mu = x.mu
    s = x.s

    (scores_indiv, scores) = main(results.T_PATH, results.GROUP, f, mu, s)
    time_end = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    y = dat()
    y.scores_indiv = scores_indiv
    y.scores = scores
    y.time_start = time_start
    y.time_end = time_end
    y.description = 'Combined channel model scores\nData file: %s\nGroup: %s\nModel file:%s' % (results.T_PATH, results.GROUP, results.MODEL)
    y.index = 'scores_indiv: Channel-Time-Frequency scores\nscores: One score per sample'
    y.save(results.S_PATH)
