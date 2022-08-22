#!/usr/bin/env python

import h5py
import numpy as np
from mctfa.utils import sigUtils


def main(data_file, group, f, mu, s, t_int=2.0, notch=[(60, 4), (120, 4),
         (180, 4), (240, 4), (80, 4), (200, 4), (220, 4)]):
    #
    #
    #
    #

    spec_options = {'norm': True, 'pruneZ': False, 'remDC': True,
                    'nperseg': t_int, 'noverlap': float(t_int)/2.}
    test_dat = h5py.File(data_file, 'r')

    (x, f) = sigUtils.compute_spectra_mult_fs(
                test_dat[group], 500, notch=notch, spec_options=spec_options)
    (E, N, S, F) = x.shape
    x_min = min([i for i in x.flatten() if i > 0])
    x = np.transpose(x, (0, 2, 1, 3))
    x = np.log(x + 0.01*x_min)

    mu_x = np.dot(np.ones((E, 1)), mu[:, :, None, :])
    s_x = np.dot(np.ones((E, 1)), s[:, :, None, :])
    y0 = -0.5*np.log(2*np.pi) - 0.5*np.log(s_x)
    y1 = -0.5*((x - mu_x)**2)/s_x
    scores_indiv = y0 + y1
    scores = np.nansum(scores_indiv, axis=(1, 2, 3))/(N*S*F)

    test_dat.close()

    return (scores_indiv, scores)


if __name__ == '__main__':
    import argparse
    import datetime
    import json
    from mctfa.data.results import dat

    time_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    parser = argparse.ArgumentParser(
                description='Independent channel and time model scoring')
    parser.add_argument(action='store', dest='T_PATH',
                        help='Path to data file')
    parser.add_argument(action='store', dest='GROUP',
                        help='HDF5 group in the data file to be used')
    parser.add_argument(action='store', dest='T_INT',
                        help='Time window in seconds')
    parser.add_argument(action='store', dest='MODEL',
                        help='Path to saved model')
    parser.add_argument(action='store', dest='S_PATH',
                        help='Path to save file')
    parser.add_argument('-n', action='store', dest='NOTCH_PATH',
                        help='Path to notch file')
    results = parser.parse_args()

    if results.NOTCH_PATH is not None:
        with open(results.NOTCH_PATH, 'r') as fid:
            notch = json.load(fid)
    else:
        notch = [(60, 4), (120, 4), (180, 4), (240, 4), (80, 4),
                 (200, 4), (220, 4)]

    t_int = float(results.T_INT)

    x = dat()
    x.load(results.MODEL)
    f = x.f
    mu = x.mu
    s = x.s

    (scores_indiv, scores) = main(
        results.T_PATH, results.GROUP, f, mu, s, t_int=t_int, notch=notch)
    time_end = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    y = dat()
    y.scores_indiv = scores_indiv
    y.scores = scores
    y.description = 'Independent channel and time model scores\nData file: %s\nGroup: %s\nTime interval: %s\nNotch filters: %s\nModel file:%s\n\nMembers\nscores_indiv: Channel-Time-Frequency scores\nscores: One score per sample\n\nStart time: %s\nEnd time: %s' % (results.T_PATH, results.GROUP, t_int, notch, results.MODEL, time_start, time_end)
    y.save(results.S_PATH)
