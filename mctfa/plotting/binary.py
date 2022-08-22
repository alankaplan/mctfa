#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def main(x, x_add_inf, s_pre):
    # Main ROC curve
    plt.figure(figsize=(10, 9))
    plt.plot(x.fpr, x.tpr, 'k', lw=2)
    plt.grid()
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('FPR', fontsize=26)
    plt.ylabel('TPR', fontsize=26)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot([0, 1], [1, 0], 'k--', lw=2)

    if x_add_inf is not None:
        plt.plot(x_add_inf.chance_1std[0], x_add_inf.chance_1std[1], 'g', lw=1)
        plt.plot(x_add_inf.chance_2std[0], x_add_inf.chance_2std[1], 'g', lw=1)

    plt.savefig(s_pre + '_roc.pdf')

    # Error rate curves
    plt.close('all')
    plt.figure(figsize=(10, 9))
    plt.plot(x.thresh, x.fpr, lw=2, label='False Positive Rate')
    plt.plot(x.thresh, 1 - x.tpr, lw=2, label='Miss Rate')
    plt.grid()
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Threshold', fontsize=26)
    plt.legend(loc='best', fontsize=24)
    plt.savefig(s_pre + '_err.pdf')

    # Score distributions
    plt.close('all')
    plt.figure(figsize=(10, 9))
    plt.plot(x.t, x.s0, 'r', lw=2, label='Negative Scores')
    plt.plot(x.t, x.s1, 'b', lw=2, label='Positive Scores')
    plt.grid()
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Score', fontsize=26)
    plt.legend(loc='best', fontsize=24)
    plt.savefig(s_pre + '_scdist.pdf')

    if 'truth_training' in dir(x_add_inf):
        # Event timeline
        i0 = np.where(np.array(x_add_inf.truth_training) == 0)[0]
        i1 = np.where(np.array(x_add_inf.truth_training) == 1)[0]
        train_pos_times = np.array(x_add_inf.train_event_times)[i1]
        train_neg_times = np.array(x_add_inf.train_event_times)[i0]
        i0 = np.where(np.array(x_add_inf.truth_testing) == 0)[0]
        i1 = np.where(np.array(x_add_inf.truth_testing) == 1)[0]
        test_pos_times = np.array(x_add_inf.test_event_times)[i1]
        test_neg_times = np.array(x_add_inf.test_event_times)[i0]
        results_pos = np.where(np.array(x.eer_event_perf)[i1] == False)[0]
        results_neg = np.where(np.array(x.eer_event_perf)[i0] == False)[0]
        min_time = x_add_inf.train_event_times[0]
        splt = (x_add_inf.train_event_times[-1] + x_add_inf.test_event_times[0] - 2*min_time)*0.5/60./60.

        plt.close('all')
        plt.figure(figsize=(10, 9))
        plt.plot((train_pos_times - min_time)/60./60., range(1, len(train_pos_times) + 1), 'b.-', lw=2, ms=12, label='Positive')
        plt.plot((train_neg_times - min_time)/60./60., range(1, len(train_neg_times) + 1), 'r.-', lw=2, ms=12, label='Negative')
        range_pos = np.array(range(len(train_pos_times) + 1, len(train_pos_times) + len(test_pos_times) + 1))
        range_neg = np.array(range(len(train_neg_times) + 1, len(train_neg_times) + len(test_neg_times) + 1))
        plt.plot((test_pos_times - min_time)/60./60., range_pos, 'b.-', lw=2, ms=12)
        plt.plot((test_neg_times - min_time)/60./60., range_neg, 'r.-', lw=2, ms=12)
        plt.plot((test_pos_times[results_pos] - min_time)/60./60., range_pos[results_pos], 'bx', ms=12, mew=4)
        plt.plot((test_neg_times[results_neg] - min_time)/60./60., range_neg[results_neg], 'rx', ms=12, mew=4)
        plt.plot([splt, splt], [0, 50], 'k--', lw=2)
        plt.grid()
        plt.legend(loc='best', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Time (hrs)', fontsize=26)
        plt.ylabel('Cumulative #', fontsize=26)
        plt.savefig(s_pre + '_event.pdf')


if __name__ == '__main__':
    import argparse
    from mctfa.data.results import dat

    parser = argparse.ArgumentParser(
                description='Binary classification visualization')
    parser.add_argument(action='store', dest='P_PATH',
                        help='Path to performance data')
    parser.add_argument('-a', action='store', dest='additional_path',
                        help='Path to additional information', default=None)
    parser.add_argument(action='store', dest='S_PRE', help='Output prefix')
    results = parser.parse_args()

    x = dat()
    x.load(results.P_PATH)
    if results.additional_path is not None:
        x_add_inf = dat()
        x_add_inf.load(results.additional_path)
    else:
        x_add_inf = None

    main(x, x_add_inf, results.S_PRE)
