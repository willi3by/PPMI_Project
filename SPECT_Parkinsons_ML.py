# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import os
import nibabel as nib
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from dltk.io.preprocessing import *
import matplotlib.pyplot as plt

#%%
os.chdir('/Users/bradywilliamson/Desktop/Parkinsons_SPECT_Project/')
base_path = '/Users/bradywilliamson/Desktop/Parkinsons_SPECT_Project/data/PPMI/'


#%% Loading, preprocessing, and writing to TFRecord.

demos = pd.read_csv('data/SPECT_ML_8_27_2019.csv')

df_GroupNum = pd.get_dummies(demos['Group'])
demos = pd.concat([demos, df_GroupNum], axis=1)
demos = demos.drop("Control", axis=1)
demos["Age_cat"] = np.ceil(demos["Age"] / 10)
train_demos, test_demos = train_test_split(demos, test_size=0.2, random_state=42, stratify=demos[['Age_cat', 'Group']])

for set_ in (train_demos, test_demos):
    set_.drop("Age_cat", axis=1, inplace=True)


#%%Get imgs for visualization.
iterator = dataset.make_one_shot_iterator()
batch_features, batch_labels = iterator.get_next()
nx = iterator.get_next()
with tf.Session() as sess:
    batch_imgs, batch_lbls = sess.run(nx)


#%%Visualize sample batch and labels.
input_tensor_shape = batch_features.shape
center_slices = [s//2 for s in input_tensor_shape]

f, axarr = plt.subplots(1, input_tensor_shape[0], figsize=(15,5))
f.suptitle('Visualization of feat batch input tensor with shape={}'.format(input_tensor_shape))

for batch_id in range(input_tensor_shape[0]):
    img_slice_ = np.squeeze(batch_features[batch_id, center_slices[1],:,:,:])
    img_slice_ = np.flip(img_slice_, axis=0)

    axarr[batch_id].imshow(img_slice_, cmap='gray');
    axarr[batch_id].axis('off')
    axarr[batch_id].set_title('Group={}'.format(batch_labels[batch_id]))

f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
plt.show();

