# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

data_path = "./data/train2/"
train_annota_path = "./data/train_annotation.txt"
val_annota_path = "./data/val_annotation.txt"

def annota_pair_gen (data_path):
  '''Split train and validation set, and generate corresponding annotaion
  Annotaion generation:
    refer: https://blog.csdn.net/zhongzhongzhen/article/details/79160872
  '''
  data_list = []
  label_list = []
  # extract data path
  for file in os.listdir(data_path):
      file_path = os.path.join(data_path, file)
      if os.path.isfile(file_path):
          data_list.append(file_path)
  num = len(data_list)
  pc = 0
  nc = 0
  for i in range(num):
    # get filename before suffix
    label = os.path.splitext(data_list[i])[0][-1]
    label_list.append(label)
    # count P/N nums
    if label == '1':
      pc += 1
    else:
      nc += 1
  assert len(data_list) == len(label_list)
  print("positive_nums:", pc, "negative_nums:", nc)  
  # split train/val, refer: https://blog.csdn.net/m0_38061927/article/details/76180541
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  # NOTE: for round times depend on n_splits
  dataset_list_split = []
  for train_indices, val_indices in split.split(data_list, label_list):
    for indices in [train_indices, val_indices]:
      print(indices, type(indices))
      dataset_list = [np.array(data_list)[indices], np.array(label_list)[indices]]
      dataset_list_split.append(dataset_list)
  print(len(dataset_list_split))
  # write into txt file
  for path, dataset_pack_list in zip([train_annota_path, val_annota_path], dataset_list_split):
    with open(path, 'w+' ,encoding='utf-8') as f:
      for d, l in zip(*dataset_pack_list):
        f.writelines([d, ' ', l, '\n'])
    
if __name__ == '__main__':
    annota_pair_gen(data_path)
