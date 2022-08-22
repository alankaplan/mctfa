import numpy as np
import h5py
import scipy.signal

x = np.random.randn(20, 500*20, 100)
y = np.random.randn(20, 500*20, 100)
z = np.random.randn(80, 500*20, 100)

noise_x = np.random.randn(20, 500*20, 100)
noise_y = np.random.randn(20, 500*20, 100)
noise_z = np.random.randn(80, 500*20, 100)


def filtbank_event(x_e, filt, ch=49):
    x_filt = x_e
    x_filt[:, ch] = scipy.signal.sosfilt(filt, x_e[:, ch])
    return x_filt


def filtbank(x, filt, ch=49):
    x_filt = []
    for xi in x:
        x_filt.append(filtbank_event(xi, filt, ch))
    return np.array(x_filt)


filt_A = scipy.signal.butter(10, [0.1, 0.2], btype='bandpass', output='sos')
filt_B = scipy.signal.butter(10, [0.2, 0.3], btype='bandpass', output='sos')

x_filt = filtbank(x, filt_A) + noise_x
y_filt = filtbank(y, filt_B) + noise_y

with h5py.File('data_train.hdf5', 'w') as f:
    dsetA = f.create_dataset('groupA', data=x_filt, shape=x_filt.shape)
    dsetB = f.create_dataset('groupB', data=y_filt, shape=y_filt.shape)

truth = np.random.randint(2, size=(80,))
z_filt = []
for (c0, i0) in enumerate(truth):
    if i0 == 0:
        filtc = filt_A
    else:
        filtc = filt_B

    zi = z[c0]
    z_filt.append(filtbank_event(zi, filtc))
z_filt = np.array(z_filt) + noise_z

with h5py.File('data_test.hdf5', 'w') as f:
    dset = f.create_dataset('test', data=z_filt, shape=z_filt.shape)

np.savetxt('truth.txt', truth, fmt='%i', delimiter='\n')
