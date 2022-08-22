import numpy as np


def chance(num_pos=18., num_neg=23.):
    t = np.linspace(0, 1, 1000)
    u_pos = t + np.sqrt(t*(1 - t)/num_pos)
    l_pos = t - np.sqrt(t*(1 - t)/num_pos)
    u_neg = t + np.sqrt(t*(1 - t)/num_neg)
    l_neg = t - np.sqrt(t*(1 - t)/num_neg)

    u_pos[np.where(u_pos > 1)[0]] = 1
    u_pos[np.where(u_pos < 0)[0]] = 0
    l_pos[np.where(l_pos > 1)[0]] = 1
    l_pos[np.where(l_pos < 0)[0]] = 0
    u_neg[np.where(u_neg > 1)[0]] = 1
    u_neg[np.where(u_neg < 0)[0]] = 0
    l_neg[np.where(l_neg > 1)[0]] = 1
    l_neg[np.where(l_neg < 0)[0]] = 0

    l_neg_1std = l_neg
    u_pos_1std = u_pos

    u_pos = t + 2*np.sqrt(t*(1 - t)/num_pos)
    l_pos = t - 2*np.sqrt(t*(1 - t)/num_pos)
    u_neg = t + 2*np.sqrt(t*(1 - t)/num_neg)
    l_neg = t - 2*np.sqrt(t*(1 - t)/num_neg)

    u_pos[np.where(u_pos > 1)[0]] = 1
    u_pos[np.where(u_pos < 0)[0]] = 0
    l_pos[np.where(l_pos > 1)[0]] = 1
    l_pos[np.where(l_pos < 0)[0]] = 0
    u_neg[np.where(u_neg > 1)[0]] = 1
    u_neg[np.where(u_neg < 0)[0]] = 0
    l_neg[np.where(l_neg > 1)[0]] = 1
    l_neg[np.where(l_neg < 0)[0]] = 0

    l_neg_2std = l_neg
    u_pos_2std = u_pos

    return (l_neg_1std, u_pos_1std, l_neg_2std, u_pos_2std)
