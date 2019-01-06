# coding: utf-8
# import os
import aifc
import numpy as np
import scipy # signal processing module
import skimage # hist equalization module
import pywt # wavelet processing module
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data_path = os.path.join("data")
# data_train_path = data_path.join("train")
# data_test_path = data_path.join("test")
# files = os.listdir(data_path)

n_frames = 4000 # read next nframes (nframes: for 1 channel and PCM audio, 1 audio frame = 1 sample = 16bits)
# STFT
n_perseg = 256 # STFT window size 128ms
n_overlap = n_perseg//2 # 50% overlap of STFT window
fs = 8e2 # audio sample rate 2kHZ, but we set STFT 800HZ (up-call range 50HZ~250HZ)
stft_window = 'hann' # STFT window type
# Wiener Filter
wiener_wind_size = (5,5) # wiener filter window size

def read_aifc(filename):
  with aifc.open(filename, 'r') as s:
    strsig = s.readframes(n_frames)
    data = np.fromstring(strsig, np.short()).byteswap()
  return np.float64(data)

def normalize_spectr_1(spectrogram):
  # TODO: Softmax_normalize
  # non_linear_normalize: xn = (x - x_mean) / max(x_max - x_mean, x_mean - x_min)
  # return np.array(
            # [(per_freq - per_freq.mean())/max((per_freq.max() - per_freq.mean()), (per_freq.mean() - per_freq.min()))
              # for per_freq in spectrogram])
  # linear_normalize: xn = (x - x_mean) / (x_max - x_min)
  return np.array(
            [(per_freq - per_freq.mean())/(per_freq.max() - per_freq.min())
              for per_freq in spectrogram])

def normalize_spectr_2(spectrogram):
  max = np.max(spectrogram.flatten())
  min = np.min(spectrogram.flatten())
  return np.array(
            [(per_freq - per_freq.mean())/per_freq.std()
              for per_freq in spectrogram])

def normalize_zero_one(spectrogram):
  max = np.max(spectrogram.flatten())
  min = np.min(spectrogram.flatten())
  return np.array(
            [(per_freq-min)/(max-min)
              for per_freq in spectrogram])
              
# TODO: shuffle func
'''load data'''
wave = read_aifc('./data/train2/20090328_000000_236s4ms_TRAIN25_1.aif')
print(np.ndarray.max(wave))
print(wave)
print(wave.shape)

'''stft'''
f, t, stft_spectr = scipy.signal.stft(wave, fs, window = stft_window, 
                        nperseg = n_perseg, noverlap = n_overlap)
                        
print(stft_spectr.shape)
stft_spectr_magn = np.abs(stft_spectr)
print(np.ndarray.max(stft_spectr_magn))

# TODO: plot3d moudle func
fig = plt.figure('3d1')
x, y = np.meshgrid(t, f)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, stft_spectr_magn)

plt.figure("spectr")
plt.subplot(131)
plt.pcolormesh(t, f, stft_spectr_magn) # TODO: np.abs(), vmin=0, vmax=1024
print(t)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

'''wiener filter'''
plt.subplot(132)
wienered_spectr = scipy.signal.wiener(stft_spectr_magn, mysize = wiener_wind_size)
print(wienered_spectr.shape)
print(np.ndarray.max(np.abs(wienered_spectr)))
plt.pcolormesh(t, f, wienered_spectr)
print(wienered_spectr)
plt.title('Wienered')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

'''normalization'''
plt.subplot(133)
normalized_spectr1 = normalize_spectr_1(wienered_spectr)
plt.pcolormesh(t, f, normalized_spectr1)
plt.title('Normalized')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

hist = skimage.exposure.histogram(normalized_spectr1, nbins=256)
print(hist)
plt.figure("spectr_hist")
spectr_flat = normalized_spectr1.flatten()
n, bins, patches = plt.hist(spectr_flat, bins=256, density=1,edgecolor='None',facecolor='red') 
print('max:', np.max(np.abs(spectr_flat)), 'std: ', spectr_flat.std())

'''equalization'''
plt.figure("equalization")
plt.subplot(121)
# normalized_spectr2 = normalize_spectr_2(wienered_spectr)
plt.pcolormesh(t, f, normalized_spectr1)
plt.title('Normalized')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.subplot(122)
print('-1_1_max:', np.max(np.abs(normalized_spectr1)), '0_1_std: ', normalized_spectr1.std())
spectr_eq1 = skimage.exposure.equalize_adapthist(normalized_spectr1)
plt.pcolormesh(t, f, spectr_eq1)
plt.title('Equalized')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

fig = plt.figure('3d2')
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(x, y, normalized_spectr1)

'''wavelet'''
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(normalized_spectr1, 'haar')
LL, (LH, HL, HH) = coeffs2
# print('wave: ', np.max(LL), LL.std())
fig = plt.figure('wavelet')
for i, a in enumerate([LL, LH, HL, HH]):
  ax = fig.add_subplot(1, 4, i + 1)
  ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
  ax.set_title(titles[i], fontsize=10)
  ax.set_xticks([])
  ax.set_yticks([])

LH_eq = skimage.exposure.equalize_hist(LH)
HL_eq = skimage.exposure.equalize_hist(HL)

spectr_eq2 = pywt.idwt2((LL, (LH_eq, HL_eq, HH)), 'haar')
fig = plt.figure('3d3')
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(x, y, normalized_spectr1)

print(np.shape(spectr_eq2[:-1,:-1]))
plt.figure('eq')
plt.subplot(121)
plt.pcolormesh(t, f, spectr_eq2[:-1,:-1])
plt.title('Equalized')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.subplot(122)
spectr_eq2_flat = spectr_eq2.flatten()
n, bins, patches = plt.hist(spectr_eq2_flat, bins=256, density=1,edgecolor='None',facecolor='red') 

plt.show()