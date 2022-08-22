#!/usr/bin/env python

import h5py
import numpy as np
from mctfa.utils import sigUtils


def compute_params(dat, notch, spec_options, fs=500):
    if len(dat.shape) == 2:
        dat = [dat]
    (x, f) = sigUtils.compute_spectra_mult_fs(dat, fs, notch=notch,
                                              spec_options=spec_options)
    (E, N, S, F) = x.shape
    x_min = min([i for i in x.flatten() if i > 0])
    x = np.transpose(x, (0, 2, 1, 3))
    x = np.log(np.reshape(x, (E*S, N, F)) + 0.01*x_min)

    mu = np.nansum(x, axis=0)
    s = np.nansum(x*x, axis=0)

    cnt = np.sum(np.isfinite(x), axis=0)

    return (f, mu, s, cnt)


def main(train_file, group, t_int=2.0,
         notch=[(60, 4), (120, 4), (180, 4), (240, 4),
                (80, 4), (200, 4), (220, 4)]):
    #
    #
    #
    #

    spec_options = {'norm': True, 'pruneZ': False, 'remDC': True,
                    'nperseg': t_int, 'noverlap': float(t_int)/2.}
    train_dat = h5py.File(train_file, 'r')

    try:
        grps = train_dat[group].keys()
        dat = train_dat[group]
    except:
        dat = {'1': train_dat[group]}
        grps = ['1']

    cnt = None
    for sub_group in grps:
        (f, mu_curr, s_curr, cnt_curr) = compute_params(dat[sub_group], notch,
                                                        spec_options, fs=500)
        if cnt is None:
            cnt = cnt_curr
            mu = mu_curr
            s = s_curr
        else:
            cnt = cnt + cnt_curr
            mu = mu + mu_curr
            s = s + s_curr

    mu = mu/cnt
    s = s/(cnt - 1) - mu**2
    s[np.where(np.isinf(s))[0]] = 0

    train_dat.close()

    return (f, mu, s)


if __name__ == '__main__':
    import argparse
    import datetime
    import json
    from mctfa.data.results import dat

    time_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    parser = argparse.ArgumentParser(
                description='Independent channel model training')
    parser.add_argument(action='store', dest='T_PATH',
                        help='Path to training data file')
    parser.add_argument(action='store', dest='GROUP',
                        help='HDF5 group in the training file to be used')
    parser.add_argument(action='store', dest='T_INT',
                        help='Time window in seconds')
    parser.add_argument(action='store', dest='S_PATH',
                        help='Path to save file')
    parser.add_argument('-n', action='store', dest='NOTCH_PATH',
                        help='Path to notch file')
    results = parser.parse_args()

    if results.NOTCH_PATH is not None:
        with open(results.NOTCH_PATH, 'r') as fid:
            notch = json.load(fid)
    else:
        notch = [(60, 4), (120, 4), (180, 4), (240, 4),
                 (80, 4), (200, 4), (220, 4)]

    t_int = float(results.T_INT)

    (f, mu, s) = main(results.T_PATH, results.GROUP, notch=notch, t_int=t_int)
    time_end = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    x = dat()
    x.f = f
    x.mu = mu
    x.s = s
    x.description = 'Independent channel model\nTraining file: %s\nGroup: %s\nTime interval: %s\nNotch filters: %s\n\nMembers\nf: Frequencies (Hz)\nmu: Mean parameters\ns: Variance parameters\n\nStart time: %s\nEnd time: %s' % (results.T_PATH, results.GROUP, t_int, notch, time_start, time_end)
    x.save(results.S_PATH)
