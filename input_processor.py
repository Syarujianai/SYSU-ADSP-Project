# coding: utf-8
import os
import aifc
import numpy as np
import scipy # signal processing module
import skimage # hist equalization module
import pywt # wavelet processing module
import tensorflow as tf
import multiprocessing as mt
import cv2

# Testing
# from train import FLAGS
FLAGS = tf.flags.FLAGS

# Read next nframes (nframes: for 1 channel and PCM audio, 1 audio frame = 1 sample = 16bits)
N_FRAMES = 4000
'''STFT'''
# STFT window size 128ms
N_PERSEG = 256
# 50% overlap of STFT window
N_OVERLAP = N_PERSEG//2
# Audio sample rate 2kHZ, but we set STFT 800HZ (up-call range 50HZ~250HZ)
FS = 8e2
# STFT window type
STFT_WIN_TYPE = 'hann'
'''Wiener Filter'''
# Wiener filter window size
WIENER_WIN_SIZE = (5,5)


def _read_aifc(filename):
  # transform b'' to str because of tf.constant()
  filename_decoded = filename.decode('utf-8', errors='ignore')
  # if file is not found, raise error
  if os.path.isfile(filename_decoded) == False:
    raise ValueError('Illegal path: %s ' % filename)
  with aifc.open(filename_decoded, 'r') as s:
    strsig = s.readframes(N_FRAMES)
    data = np.fromstring(strsig, np.short()).byteswap()
  return np.float64(data)


def _normalize_spectr_1(spectrogram):
  return np.array(
            [(per_freq - per_freq.mean())/(per_freq.max() - per_freq.min())
              for per_freq in spectrogram])


def _normalize_spectr_2(spectrogram):
  max = np.max(spectrogram.flatten())
  min = np.min(spectrogram.flatten())
  return np.array(
            [(per_freq - per_freq.mean())/per_freq.std()
              for per_freq in spectrogram])


def _normalize_zero_one(spectrogram):
  max = np.max(spectrogram.flatten())
  min = np.min(spectrogram.flatten())
  return np.array(
            [(per_freq-min)/(max-min)
              for per_freq in spectrogram])

              
def _read_and_preprocess(filename, label):
  '''Encapsulation python function for tf.py_func() 
  ''' 
  # load data
  wave = _read_aifc(filename)
  # stft
  f, t, stft_spectr = scipy.signal.stft(
                        wave, FS, window=STFT_WIN_TYPE, 
                        nperseg=N_PERSEG, noverlap=N_OVERLAP)                        
  stft_spectr_magn = np.abs(stft_spectr)
  # normalization
  normalized_spectr = _normalize_spectr_1(stft_spectr_magn)
  # wiener filter
  wienered_spectr = scipy.signal.wiener(normalized_spectr, mysize=WIENER_WIN_SIZE)
  # equalization
  # spectr_equalized = skimage.exposure.equalize_adapthist(normalized_spectr)
  # wavelet soft threshold filter
  soft_spectr = pywt.threshold(wienered_spectr, 0.15, 'hard',0)
  # spectr_resized = skimage.transform.resize(spectr_equalized, [129,33])
  # cv.resize() faster than skimage.transform.resize()?
  spectr_resized = cv2.resize(soft_spectr, (129,33), interpolation=cv2.INTER_LINEAR)
  
  return spectr_resized, label


def input_process_fn_builder(path):
  '''Input Process Fnction Builder
  Rerturn:
    steps_per_epoch, input_process_fn()
  Description:
    decoupling by repeate some code for tf.estimator could accept only one return value 'dataset' input_fn
  '''
  # read (data, label) pairs as list
  pairs_list = np.loadtxt(path, dtype = str).tolist()
  # shuffle pairs list
  np.random.shuffle(pairs_list)
  # shuffled list
  data_list, label_list = zip(*[(p[0], int(p[1])) for p in pairs_list])
  # steps per epoch
  steps_per_epoch = np.ceil(len(label_list) / FLAGS.TRAIN_BATCH_SIZE).astype(np.int32)
  def input_process_fn(path, batch_size):
    ''' Data Preprocessing 
    tf.data framework:
      refer: https://blog.csdn.net/DumpDoctorWang/article/details/84028957
    '''
    # read (data, label) pairs as list
    pairs_list = np.loadtxt(path, dtype = str).tolist()
    # shuffle pairs list
    np.random.shuffle(pairs_list)
    # shuffled list
    data_list, label_list = zip(*[(p[0], int(p[1])) for p in pairs_list])
    # instantiate dataset with (filename_list, label_list)
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(data_list), tf.constant(label_list)))
    # data pipeline
    # refer: https://www.tensorflow.org/guide/datasets?hl=zh-cn
    # _read_and_preprocess(filename, label) -> return: (tf.float64, label.dtype)
    dataset = dataset.map(
                lambda filename, label: tuple(tf.py_func(
                        _read_and_preprocess, [filename, label], [tf.float64, label.dtype])),
                        num_parallel_calls=mt.cpu_count())
    # set training batch size
    dataset = dataset.batch(batch_size)
    # repeate dataset
    dataset = dataset.repeat()
    return dataset
    
  return input_process_fn, steps_per_epoch


def inputs_test():
  '''Test input_process_fn() Function
  '''
  input_fn, steps = input_process_fn_builder(FLAGS.TRAIN_ANNOTA_PATH)
  dataset = input_fn(FLAGS.TRAIN_ANNOTA_PATH, FLAGS.TRAIN_BATCH_SIZE)
  print('shapes:', dataset.output_shapes)
  print('types:', dataset.output_types)
  print('steps:', steps)
  # NOTE: tf.estimator only support one_shot_iterator
  next_op = dataset.make_one_shot_iterator().get_next()

  with tf.Session() as sess:
    for i in range(5):
      data, label = sess.run(next_op)
      print(len(data), len(label), data.shape, np.min(data), np.max(data))


def main(unused_argv):
  pass
  
  
if __name__ == '__main__':
    tf.app.run(inputs_test())
