#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:33:17 2019

@author: bradywilliamson
"""


import tensorflow as tf
import numpy as np
from glob import glob
import subprocess
from dltk.io.preprocessing import *
from dltk.io.augmentation import *
import time
import ants
#%%

def load_img(subj_id, base_path):
    x = []

    subj_id_str = str(subj_id)
    # subj_path = base_path + subj_id_str + '/Reconstructed_DaTSCAN/**/*.dcm'
    # data_path = glob(subj_path, recursive=True)
    ##Convert to nifti.
    # def bash_command(cmd):
    #     subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # cmd = ['dcm2niix', data_path[0]]
    # bash_command(cmd)
    # time.sleep(5)
    
    glob_nii_path = base_path + subj_id_str + '/Reconstructed_DaTSCAN/**/*.nii'
    nii_path = glob(glob_nii_path, recursive=True)
    print(subj_id)
    img = ants.image_read(nii_path[0])
    img = img.numpy()
    SPECT = normalise_zero_one(img)
    SPECT = SPECT[..., np.newaxis]

    return SPECT

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_TFRecord(filename, demos, base_path):

    writer = tf.io.TFRecordWriter(filename)
    subjs = list(demos["Subject"])
    for subject in subjs:
        label = np.array(demos.loc[demos['Subject'] == subject, "PD"])[0]
        img = load_img(subject, base_path)
        feature = {'train/label': _float_feature(label.ravel()),
                   'train/image': _float_feature(img.ravel())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

def parse_record(raw_record):
    keys_to_features = {
            'train/image': tf.io.FixedLenFeature([91,109,91,1], tf.float32),
            'train/label': tf.io.FixedLenFeature([], tf.int64)}

    parsed = tf.io.parse_single_example(raw_record, keys_to_features)

    return parsed['train/image'], parsed['train/label']

def input_fn(is_training, train_filename, batch_size, num_epochs=1, num_parallel_calls=1):
    # tf.compat.v1.disable_eager_execution()
    dataset = tf.data.TFRecordDataset(train_filename).map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=64)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def train_input_fn(file_path, batch_size, num_epochs):
    return input_fn(True, file_path, batch_size, num_epochs, 10)

def validation_input_fn(file_path, batch_size):
    return input_fn(False, file_path, batch_size, 1, 1)


