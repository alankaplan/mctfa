#!/usr/bin/env python

import sklearn.metrics
import scipy.stats
import sys
import numpy as np
import datetime
import scipy.interpolate


def tpr_at(fpr, tpr, fpr_val):
    tpr_int = np.interp(fpr_val, fpr, tpr)

    return tpr_int


def get_eer(fpr, tpr, thresh):
    fpr_int = scipy.interpolate.interp1d(thresh, fpr)
    tpr_int = scipy.interpolate.interp1d(thresh, tpr)
    t = np.linspace(min(thresh), max(thresh), 100000)
    i0 = np.argmin(np.abs(fpr_int(t) - 1 + tpr_int(t)))
    eer0 = fpr_int(t)[i0]
    eer1 = 1 - tpr_int(t)[i0]

    if np.abs(eer0 - eer1) > 0.0001:
        sys.stdout.write('EER diff\n')

    return (t[i0], 0.5*eer0 + 0.5*eer1)


def sc_dist(sc, truth):
    i0 = np.where(truth == 0)[0]
    i1 = np.where(truth == 1)[0]

    k0 = scipy.stats.gaussian_kde(sc[i0])
    k1 = scipy.stats.gaussian_kde(sc[i1])

    t = np.linspace(min(sc) - 3*np.std(sc), max(sc) + 3*np.std(sc), 10000)

    s0 = k0(t)
    s1 = k1(t)

    return (t, s0, s1)


def discr_comps(sc0, truth):
    i0 = np.where(truth == 0)[0]
    i1 = np.where(truth == 1)[0]

    # Frequency bin
    x = np.nanmean(sc0, axis=(1, 2))
    s1 = np.nanmean(x[i1], axis=0)
    s0 = np.nanmean(x[i0], axis=0)
    d_freq = s1 - s0

    # Channel
    x = np.nanmean(sc0, axis=(2, 3))
    s1 = np.nanmean(x[i1], axis=0)
    s0 = np.nanmean(x[i0], axis=0)
    d_ch = s1 - s0

    # Time bin
    x = np.nanmean(sc0, axis=(1, 3))
    s1 = np.nanmean(x[i1], axis=0)
    s0 = np.nanmean(x[i0], axis=0)
    d_time = s1 - s0

    return (d_freq, d_ch, d_time)


def event_perf(sc, th, truth):
    n = len(truth)
    res = [truth[i0] == (sc[i0] >= th) for i0 in range(n)]
    return res


def main(truth_file, m1_scores, m0_scores, m1_scores_indiv, m0_scores_indiv,
         sc_path=None):
    truth = np.loadtxt(truth_file, delimiter='\n')

    if sc_path is None:
        sc = m1_scores - m0_scores
        sc_indiv = m1_scores_indiv - m0_scores_indiv
    else:
        sc = np.loadtxt(sc_path, delimiter=',')
        sc_indiv = None

    (fpr, tpr, thresh) = sklearn.metrics.roc_curve(truth, sc)
    (eer_th, eer) = get_eer(fpr, tpr, thresh)

    (t, s0, s1) = sc_dist(sc, truth)

    fpr_sel = np.arange(1, 51)/100.
    tpr_sel = tpr_at(fpr, tpr, fpr_sel)

    d_freq = None
    d_ch = None
    d_time = None
    if sc_indiv is not None:
        if len(sc_indiv.shape) == 4:
            (d_freq, d_ch, d_time) = discr_comps(sc_indiv, truth)

    eer_event_perf = event_perf(sc, eer_th, truth)

    return (eer_th, eer, fpr, tpr, thresh, t, s1, s0, fpr_sel, tpr_sel, d_freq, d_ch, d_time, eer_event_perf)


if __name__ == '__main__':
    import argparse
    from mctfa.data.results import dat

    time_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    parser = argparse.ArgumentParser(
                description='Binary classification performance evaulation')
    parser.add_argument(action='store', dest='T_PATH',
                        help='Path to truth labels')
    parser.add_argument(action='store', dest='M1_SCORES',
                        help='Path to model 1 scores')
    parser.add_argument(action='store', dest='M0_SCORES',
                        help='Path to model 0 scores')
    parser.add_argument(action='store', dest='S_PATH',
                        help='Path to save file')
    parser.add_argument('-s', action='store', dest='score_path',
                        help='Path to scores', default=None)
    results = parser.parse_args()

    if results.score_path is None:
        x1 = dat()
        x1.load(results.M1_SCORES)
        m1_scores = x1.scores
        m1_scores_indiv = x1.scores_indiv
        x0 = dat()
        x0.load(results.M0_SCORES)
        m0_scores = x0.scores
        m0_scores_indiv = x0.scores_indiv
    else:
        m1_scores = None
        m1_scores_indiv = None
        m0_scores = None
        m0_scores_indiv = None

    (eer_th, eer, fpr, tpr, thresh, t, s1, s0, fpr_sel, tpr_sel, d_freq, d_ch,
     d_time, eer_event_perf) = main(results.T_PATH, m1_scores=m1_scores,
                                    m0_scores=m0_scores,
                                    m1_scores_indiv=m1_scores_indiv,
                                    m0_scores_indiv=m0_scores_indiv,
                                    sc_path=results.score_path)
    time_end = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    y = dat()
    y.eer_th = eer_th
    y.eer = eer
    y.fpr = fpr
    y.tpr = tpr
    y.thresh = thresh
    y.t = t
    y.s1 = s1
    y.s0 = s0
    y.fpr_sel = fpr_sel
    y.tpr_sel = tpr_sel
    y.d_freq = d_freq
    y.d_ch = d_ch
    y.d_time = d_time
    y.eer_event_perf = eer_event_perf
    y.description = 'Performance Data\nTruth file: %s\nModel 1 scores: %s\nModel 0 scores: %s\n\nMembers\neer_th: Equal error rate threshold\neer: Equal error rate\nfpr: False positive rates\ntpr: True positive rates\nthresh: Thresholds\nt: Score values\ns1: Positive score distribution\ns0: Negative score distribution\nfpr_sel: Selected false positive values\ntpr_sel: Correspondong true positive values\nd_freq: Frequency bin discriminative scores\nd_ch: Channel discriminative scores\nd_time: Time bin discriminative scores\neer_event_perf: Per event performance operating at the EER\n\nStart time: %s\nEnd time: %s' % (results.T_PATH, results.M1_SCORES, results.M0_SCORES, time_start, time_end)
    y.save(results.S_PATH)
