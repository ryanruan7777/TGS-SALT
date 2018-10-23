# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:09:57 2018

@author: Ryan
"""

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d as montage
from skimage.morphology import binary_opening, disk, label
import data_read
import keras
import tensorflow as tf
from keras import models, layers
from resnet_unet import unet
import scipy


seg_model = unet()
weight_path="{}_weights.best.hdf5".format('seg_model')
seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')
IMG_SCALING = (1, 1)
pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(axis=0).max(), pred_y.max(axis=0).min(), pred_y.mean())
from skimage import io,data,color


fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.hist(pred_y.ravel(), np.linspace(-3, 3, 50))
ax.set_xlim(-3, 3)
ax.set_yscale('log', nonposy='clip')

#%%%
IOU1=0
IOU2=0
a=0
for i in range(600):
    pred_binary=pred_y[i,:,:,0]
    pred_binary[pred_binary>0]=1
    pred_binary[pred_binary<=0]=0
    pred_binary=np.array(pred_binary,dtype=np.uint8)
    pred_mask=color.gray2rgb(pred_binary) 
    pred_mask=scipy.misc.imresize(pred_mask,[101,101,3])
    
    valid_binary=valid_y[i,:,:,0]
    valid_mask=color.gray2rgb(valid_binary) 
    valid_mask=scipy.misc.imresize(valid_mask,[101,101,3])
    #i=(np.sum(pred_mask[:,:,0]*valid_mask[:,:,0])+np.sum((1-pred_mask[:,:,0])*(1-valid_mask[:,:,0])))/(101*101)
    if np.sum(valid_mask[:,:,0])!=0:
        a=a+1
        i=(0.00001+np.sum(pred_mask[:,:,0]*valid_mask[:,:,0]))/(0.00001+np.sum(valid_mask[:,:,0])+np.sum(pred_mask[:,:,0])-(np.sum(pred_mask[:,:,0]*valid_mask[:,:,0])))
        IOU1=IOU1+i
    j=(np.sum(pred_binary[:,:]*valid_binary[:,:])+np.sum((1-pred_binary[:,:])*(1- valid_binary[:,:])))/(96*96)
    IOU2=IOU2+j/600
IOU1=IOU1/(a)
#%%
fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.hist(pred_y.ravel(), np.linspace(0, 1, 50))
ax.set_xlim(0, 1)
ax.set_yscale('log', nonposy='clip')

if IMG_SCALING is not None:
    fullres_model = models.Sequential()
    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
    fullres_model.add(seg_model)
    fullres_model.add(layers.UpSampling2D(IMG_SCALING))
else:
    fullres_model = seg_model
fullres_model.save('fullres_model.h5')

#%%%%

def predict(img, path=test_image_dir):
    c_img = imread(os.path.join(path, (c_img_name+'.png')))
    c_img=scipy.misc.imresize(c_img,[96,96,3])
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>0.5, np.expand_dims(disk(2), -1))
    return cur_seg, c_img

#%%
## Get a sample of each group of ship count
samples = valid_df.groupby('ships').apply(lambda x: x.sample(10))
fig, m_axs = plt.subplots(samples.shape[0], 4, figsize = (15, samples.shape[0]*4))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
    first_seg, first_img = predict(c_img_name, train_image_dir)
    ax1.imshow(first_img[0])
    ax1.set_title('Image: ' + c_img_name)
    ax2.imshow(first_seg[:, :, 0])
    ax2.set_title('Model Prediction')
    '''
    reencoded = masks_as_color(multi_rle_encode(first_seg[:, :, 0]))
    ax3.imshow(reencoded)
    ax3.set_title('Prediction Re-encoded')
    '''
    ground_truth = masks_as_color(masks.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels'])
    ground_truth=ground_truth
    ax3.imshow(ground_truth)
    ax3.set_title('Ground Truth')
    
fig.savefig('validation.png')