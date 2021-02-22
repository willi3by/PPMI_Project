# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tf_keras_vis.gradcam import Gradcam
import matplotlib.pyplot as plt
from tf_keras_vis.utils import normalize
import numpy as np
from matplotlib import cm
import ants
import shutil

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    
loss = lambda output: K.mean(output[:])

#%%After building model.

def parse_record(raw_record):
    keys_to_features = {
            'train/image': tf.io.FixedLenFeature([91,109,91,1], tf.float32),
            'train/label': tf.io.FixedLenFeature([], tf.int64)}

    parsed = tf.io.parse_single_example(raw_record, keys_to_features)

    return parsed['train/image'], parsed['train/label']

dataset = tf.data.TFRecordDataset('/Users/bradywilliamson/Desktop/UC_DATscan_Project/data/UCDAT.tfrecord').map(parse_record)
dataset = dataset.batch(99)
nx = tf.compat.v1.data.make_one_shot_iterator(dataset)
x_test, y_test = nx.get_next()

model.load_weights('../model/best_model.h5')
#%%
for i in range(x_test.shape[0]):
    subject = demos['Subject'][i]
    gradcam = Gradcam(model, model_modifier)
    test_dataset = x_test[i].numpy()
    test_dataset = test_dataset[np.newaxis,...]
    cam = gradcam(loss, test_dataset)
    cam = normalize(cam)
    # heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)
    spect_template = ants.image_read('/Users/bradywilliamson/Desktop/brain_templates/DATscan_MNI_template.nii')
    affine = np.eye(4)
    img_nii = nib.Nifti1Image(np.squeeze(test_dataset), affine)
    img_nii.to_filename('tmp.nii')
    img = ants.image_read('tmp.nii')
    # img_denoise = img.denoise_image()
    img_smooth = img.smooth_image(2)
    mytx = ants.registration(img_smooth, spect_template, type='Affine')
    img_mni = ants.apply_transforms(spect_template, img_smooth, transformlist=mytx['invtransforms'])
    dat_string = '{}_dat_mni.nii'
    ants.image_write(img_mni, dat_string.format(subject))
    os.remove('tmp.nii')
    cam_img = np.squeeze(cam)
    cam_img = nib.Nifti1Image(cam_img, affine)
    cam_img.to_filename('tmp_cam.nii')
    cam_img = ants.image_read('tmp_cam.nii')
    cam_mni = ants.apply_transforms(spect_template, cam_img, transformlist=mytx['invtransforms'])
    cam_string = '{}_cam_img.nii'
    ants.image_write(cam_mni, cam_string.format(subject))
    os.remove('tmp_cam.nii')

#%%

#%%
import matplotlib.pyplot as plt
heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)
img_slice_ = np.squeeze(x_test[10, 40,:,:,:])
img_slice_ = np.flip(img_slice_, axis=0)
plt.imshow(img_slice_, cmap='gray')
heat_slice_ = np.squeeze(heatmap[0,40,...])
heat_slice_ = np.flip(heat_slice_, axis=0)
plt.imshow(heat_slice_, cmap='jet', alpha=0.3)

#%% Export NIFTI. 
spect_template = ants.image_read('/Users/bradywilliamson/Desktop/brain_templates/DATscan_MNI_template.nii')
affine = np.eye(4)
img = np.squeeze(test_dataset)
img = img.transpose(2,1,0)
img = np.flip(img, axis=1)
img = ants.from_numpy(img)
img_denoise = img.denoise_image()
img_smooth = img_denoise.smooth_image(2)
mytx = ants.registration(spect_template, img, type='Rigid')
img_mni = mytx['warpedmovout']
ants.image_write(img_mni, 'dat_mni.nii')
cam_img = np.squeeze(cam)
cam_img = cam_img.transpose(2,1,0)
cam_img = np.flip(cam_img, axis=1)
cam_img = ants.from_numpy(cam_img)
cam_mni = ants.apply_transforms(spect_template, cam_img, transformlist=mytx['fwdtransforms'])
ants.image_write(cam_mni, 'cam_mni.nii')
