# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:41:02 2019

@author: jpontalb
"""

import os
import functools
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from PIL import Image
from skimage import io
#from astropy import *
import tensorflow as tf
import numpy as np

sess = tf.Session()

# %%

img = io.imread('cat.png')
kernel = coordinates = np.random.rand(3, 3, 3)

# %%
#----------------2D---------------------#

ones_2d = img/255
weights_2d = tf.random.normal(shape = (3,3,3,1),dtype=tf.float32, seed=None,  name=None)

strides_2d = [1, 1, 1, 1]

in_2d = tf.constant(ones_2d, dtype=tf.float32)



in_width = int(in_2d.shape[0])
in_height = int(in_2d.shape[1])

input_2d   = tf.reshape(in_2d, [1, in_width,in_height, 3])

output_2d = tf.squeeze(tf.nn.conv2d(input_2d, weights_2d, strides=strides_2d, padding='SAME'))

sess = tf.Session()
with sess.as_default():
	test_c2= output_2d.eval()

output_2 = test_c2
#output_2 = (np.round(output_2)).astype('uint8')
output_2 = (output_2-np.min(output_2.ravel()))/(np.max(output_2.ravel()) - np.min(output_2.ravel()))
output_2 = np.round(output_2*255)
output_2 = output_2.astype('uint8')

# %%
#----------------3D---------------------#
#ones_3d = np.ones((5,5,5))

ones_3d = img

weights_3d = tf.random.normal(shape = (3,3,3,1,1),dtype=tf.float32, seed=None,  name=None)
strides_3d = [1, 1, 1, 1, 1]

in_3d = tf.constant(ones_3d, dtype=tf.float32)

in_width = int(in_3d.shape[0])
in_height = int(in_3d.shape[1])
in_depth = int(in_3d.shape[2])



input_3d   = tf.reshape(in_3d, [1, 3, in_width,in_height, 1])

output_3d = tf.squeeze(tf.nn.conv3d(input_3d, weights_3d, strides=strides_3d, padding='SAME'))

sess = tf.Session()
with sess.as_default():
	test= output_3d.eval()
	
	
h = test
h = h.reshape(1200,1600,3)

h = (output_2-np.min(h.ravel()))/(np.max(h.ravel()) - np.min(h.ravel()))
h = np.round(h*255)
h = h.astype('uint8')

plt.figure()
plt.imshow(img)
plt.title('Original')

plt.figure()
plt.subplot(121)
plt.imshow(output_2, cmap = 'gray')
plt.title('Output from Conv2D')
plt.subplot(122)
plt.imshow(h, cmap = 'gray')
plt.title('Output of Conv3D')
