#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:17:22 2019

@author: bradywilliamson
"""

import sklearn as skl
import tensorflow as tf
import seaborn as sns

def parse_record(raw_record):
    keys_to_features = {
            'train/image': tf.io.FixedLenFeature([91,109,91,1], tf.float32),
            'train/label': tf.io.FixedLenFeature([], tf.int64)}

    parsed = tf.io.parse_single_example(raw_record, keys_to_features)

    return parsed['train/image'], parsed['train/label']
#%%

dataset = tf.data.TFRecordDataset('../../').map(parse_record)
dataset = dataset.batch(125)
nx = tf.compat.v1.data.make_one_shot_iterator(dataset)
x_test, y_test = nx.get_next()

#%%
model_weights = '/Users/bradywilliamson/Desktop/Parkinsons_SPECT_Project/model/best_model.h5'
model.load_weights(model_weights)
y_probas = model.predict_proba(x_test)
y_pred = model.predict_classes(x_test)
#%%
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
fpr, tpr, thresholds = roc_curve(y_test, y_probas)
auc_score = auc(fpr, tpr)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
TP = confusion[1,1]
FP = confusion[0,1]
TN = confusion[0,0]
FN = confusion[1,0]

sensitivity = skl.metrics.recall_score(y_test, y_pred)
specificity = TN / (TN + FP)


#%%
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_score))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.4)
plt.ylim(0.6, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_score))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
#%%
classes = ["No PD", "PD"]
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d',xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
#%% Plot data that was misclassified.
y_test_array = y_test.numpy()
diff = np.subtract(y_test_array, y_pred.ravel())
idxs = [i for i in range(len(diff)) if diff[i] > 0]
for i in range(len(idxs)):
    input_tensor_shape = x_test.shape
    center_slices = [s//2 for s in input_tensor_shape]

    f, axarr = plt.subplots(1, len(idxs), figsize=(15,5))
    # f.suptitle('Visualization of feat batch input tensor with shape={}'.format(input_tensor_shape))
    
    for batch_id in range(len(idxs)):
        img_slice_ = np.squeeze(x_test[batch_id, :,:,40,:])
        img_slice_ = np.flip(img_slice_, axis=0)
        img_slice_ = np.rot90(img_slice_)
        plt.imshow(img_slice_, cmap='gray')
        plt.axis('off')
        plt.title("PD")
        
        
        axarr[batch_id].imshow(img_slice_, cmap='gray');
        axarr[batch_id].axis('off')
        axarr[batch_id].set_title('Group={}'.format(y_test[idxs[i]]))
        axarr[batch_id].set_title('False Negative')

# f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
plt.show();

#%%

