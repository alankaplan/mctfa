import scipy.signal
import numpy as np


def normPSD(s):
    # Computes the PSD of a normalized signal given the unnormalized PSD
    #
    # Input
    #   s: input PSD (single sided)
    #       t x f
    #
    # Output
    #   s_norm: output PSD
    #       if the input is s = PSD(x), then the output is
    #                       s_norm = PSD((x - mean(x))/std(x))
    #

    (F, T) = s.shape

    v = (np.sum(s[1:, :], axis=0)/(F**2))[None, :]

    i0 = np.where(v <= 0)[1]
    v[:, i0] = 1.0

    s_norm = s.copy()
    s_norm[0, :] = 0
    s_norm = s_norm/(np.ones((F, 1))*v)

    return s_norm


def compute_spectra_mult_fs(y, fs, notch=[],
                            spec_options={'nperseg': 1, 'noverlap': 0.5,
                                          'window': 'hann', 'norm': True}):
    x = []

    for i0 in range(len(y)):
        y_curr = y[i0]
        (x_curr, f) = compute_spectra_fs(y_curr, fs, notch=notch,
                                         **spec_options)
        x = x + [x_curr]

    return (np.array(x), f)


def compute_spectra_fs(x, fs, notch=[], nperseg=1, noverlap=0.5, window='hann',
                       norm=False, pruneZ=False, remDC=False):
    # Computes spectra from channel data
    #
    # Inputs:
    #   x:  channel data
    #           time x num_channels
    #
    # Outputs:
    #   x_spectra:  spectra
    #                   num_channels x frequency_bin x time_bin
    #   t:          time stamps
    #                   time_bin,
    #   f:          frequency stamps
    #                   freq_bin,
    #

    (T, N) = x.shape
    nperseg_samp = int(float(nperseg)*fs)
    noverlap_samp = int(float(noverlap)*fs)

    x_spectra = []
    for i0 in range(N):
        if nperseg_samp > T:
            nfft = nperseg_samp
        else:
            nfft = None
        if noverlap_samp >= T:
            noverlap_samp = T - 1   # Will only compute 1 time bin, since nperseg_samp == T
        (f, t, s) = scipy.signal.spectrogram(x[:, i0], nperseg=nperseg_samp,
                                             fs=fs, noverlap=noverlap_samp,
                                             window=window, nfft=nfft)
        if len(notch) > 0:
            for (notch_freq, notch_w) in notch:
                notch_samp = max(int(float(notch_w)/(f[1] - f[0])), 1)
                i0 = np.where(f - notch_freq >= 0)[0][0]
                s = np.delete(s, range(i0 - notch_samp, i0 + notch_samp), axis=0)
                f = np.delete(f, range(i0 - notch_samp, i0 + notch_samp))

        if norm:
            s = normPSD(s)
        x_spectra.append(s)

    (F, S) = s.shape

    x = np.transpose(x_spectra, (0, 2, 1))
    if pruneZ:
        i0 = np.where(np.sum(x, axis=2) <= 0)
        if len(i0[0]) > 0:
            x = np.delete(x, np.unique(i0[1]), axis=1)

    if remDC:               # FIXME unlikely case that 0 Hz is notched out
        x = x[:, :, 1:]
        f = f[1:]

    return (x, f)
