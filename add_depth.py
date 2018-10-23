# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:35:04 2018

@author: Ryan
"""
'''
It's a test to produce the depth_channel
'''

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
#from skimage.util.montage import montage2d as montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() # memory is tight
from keras.losses import binary_crossentropy
import tensorflow as tf
import scipy
from skimage import io,data,color
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from tqdm import tqdm_notebook #, tnrange
#import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam,SGD
import keras.backend as K
from sklearn.model_selection import StratifiedKFold 
import cv2
#%%%
#### Reference  from Heng's discussion
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)

    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical

    percentage = cover/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1,8))
    for c in coverage:
        histall[0,c] += 1
    return histall



#%%%

def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
ship_dir = r'E:/kaggle2/data'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

img_size_ori = 101
img_size_target = 101
cv_total = 5

train_df = pd.read_csv(os.path.join(ship_dir, 'train.csv'), index_col="ImageId", usecols=[0])
depth = pd.read_csv(os.path.join(ship_dir, 'depths.csv'), index_col="id")
train_df = train_df.join(depth)
test_df = depth[~depth.index.isin(train_df.index)]
len(train_df)

train_df["images"] = [np.array(load_img("../data/train/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["masks"] = [np.array(load_img("../data/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

train_df["coverage_class"] = train_df.masks.map(get_mask_type)

#depth data
train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total, random_state=1234, shuffle=True)
for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape,evaluate_index.shape) # the shape is slightly different in different

def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    x_train = np.array(train_df.images[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df.masks[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df.images[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df.masks[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_train_depth = add_channel(train_index)
    print(x_train.shape)
    print(x_train_depth.shape)
    #x_train = np.concatenate((x_train, x_train_depth), axis= -1)
    x_valid_depth = add_channel(evaluate_index)
    x_valid = np.concatenate((x_valid, x_valid_depth), axis= -1)
    return x_train,y_train,x_valid,y_valid,x_train_depth

def add_channel(index):
    depth_channle = []
    
    b = np.array(train_df.z[index])
    for i in range(b.size):
        depth = b[i]*np.ones((101,101,1))
        depth_channle.append(depth)
        
    return np.array(depth_channle)

def add_depth_channels_v2(train_np):
    
    depth_channle = []
    depth_one = np.zeros((101,101,1))
    
    for row, const in enumerate(np.linspace(0,1,101)):
        depth_one[row,:,:] = const
               
    for i in range(train_np.shape[0]):
        depth = train_np[i]*depth_one
        depth_channle.append(depth)
        
    return np.array(depth_channle)

def add_depth_channels_v3(train_np):
    depth_channle =[]
    
    border = 5
    for i in range(train_np.shape[0]):
        img_center_mean = train_np[i,border:-border,border:-border,:].mean()
        img_csum = (np.float32(train_np[i])-img_center_mean).cumsum(axis=0)
        img_csum -= img_csum[border:-border, border:-border].mean()
        img_csum /= max(1e-3, img_csum[border:-border, border:-border].std())
        
        depth_channle.append(img_csum)
        
    return np.array(depth_channle)

x_train, y_train, x_valid, y_valid, x_train_depth =  get_cv_data(1)

# =============================================================================
# x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
# x_train_depth = np.append(x_train_depth, [np.fliplr(x) for x in x_train_depth], axis=0)
# x_train = np.concatenate((x_train, x_train_depth), axis=-1)
# y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
# =============================================================================
