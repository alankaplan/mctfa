#!/usr/bin/env python

import h5py
import numpy as np
import datetime
from mctfa.utils import sigUtils


def main(train_file, group, t_int=2.0):
    #
    #
    #
    #

    spec_options = {'norm': True, 'pruneZ': False, 'remDC': True,
                    'nperseg': t_int, 'noverlap': float(t_int)/2.}
    notch = [(60, 4), (120, 4), (180, 4), (240, 4), (80, 4),
             (200, 4), (220, 4)]

    train_dat = h5py.File(train_file, 'r')

    (x, f) = sigUtils.compute_spectra_mult_fs(
                                              train_dat[group],
                                              500,
                                              notch=notch,
                                              spec_options=spec_options)
    (E, N, S, F) = x.shape
    x_min = min([i for i in x.flatten() if i > 0])
    x = np.log(np.reshape(x, (E*N*S, F)) + 0.01*x_min)

    mu = np.nanmean(x, axis=0)
    s = np.nanvar(x, axis=0)

    train_dat.close()

    return (f, mu, s)


if __name__ == '__main__':
    import argparse
    from mctfa.data.results import dat

    time_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    parser = argparse.ArgumentParser(
                description='Combined channel model training')
    parser.add_argument(
                action='store', dest='T_PATH',
                help='Path to training data file')
    parser.add_argument(
                action='store', dest='GROUP',
                help='HDF5 group in the training file to be used')
    parser.add_argument(
                action='store', dest='S_PATH',
                help='Path to save file')
    results = parser.parse_args()

    (f, mu, s) = main(results.T_PATH, results.GROUP)
    time_end = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    x = dat()
    x.f = f
    x.mu = mu
    x.s = s
    x.time_start = time_start
    x.time_end = time_end
    x.description = 'Combined channel model\nTraining file: %s\nGroup: %s' % (results.T_PATH, results.GROUP)
    x.parameters = 'f: Frequencies (Hz)\nmu: Mean parameters\ns: Variance parameters'
    x.save(results.S_PATH)
