# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:14:49 2019

@author: Justi
"""

# %% 
#---------------------General Libraries--------------------#
import os
import functools
import time
from numpy import expand_dims
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from skimage import io

import seaborn as sns

#---------------------Utility Libraries--------------------#

from utils import *
from utils import bce_dice_loss, dice_loss,dice_coeff
from unet import *
from validation_metrics_results import *

#---------------------Tensorflow Libraries--------------------#
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf
#import tensorflow.contrib as tfcontrib
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from tensorflow.image import rgb_to_grayscale
#from keras.preprocessing.image import img_to_array

# %%

model = models.load_model('F:/unet_chosen_models/weights_2019_8_23_15_33_17_GAN_weights.40-1.06.hdf5', 
                          custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})

# %%

for i in range(len(model.layers)):
    print(model.layers[i].name)

model.summary()
# %%
layer_name = 'activation_2'
intermediate_model = intermediateLayer(model, layer_name )

# %%

un_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/UN/images/'
cn_base_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/'
save_path = "G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/"
method_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/'

# %%

un_dir = os.listdir(un_path)
method_path_dir = os.listdir(method_path)

acti = []
avg_jen = []
category = []
img_list = []
avg_name = []
n = []

for method in method_path_dir:
    
    cn_path = cn_base_path + method + '/images/'
    cn_dir = os.listdir(cn_path)
    
    for path1,path2 in zip(un_dir,cn_dir):
        
        print('Predicting...')
        img = io.imread(un_path + path1)
    
        img_UN = expand_dims(img,axis = 0)
        img_UN = img_UN/255
        
        img_gray = color.rgb2gray(img)
        img_gray = np.expand_dims(img_gray,axis = 0)
        img_gray = np.stack([img_gray,img_gray,img_gray], 3)
        
        img_CN = io.imread(cn_path + path2)
        img_CN = expand_dims(img_CN,axis = 0)
        img_CN = img_CN/255
            
        feature_maps_gray = intermediate_model.predict(img_gray)
        feature_maps_cn = intermediate_model.predict(img_CN)
        feature_maps_UN = intermediate_model.predict(img_UN)
        
        acti.extend(activationMetrics(feature_maps_cn, feature_maps_UN))
        img_name = path1
        check_shape = np.shape(feature_maps_UN)
        shape = check_shape[3]
        for j in range(shape):
            category.append(method)
            img_list.append(img_name)
            
        avg_jen.append(np.mean(activationMetrics(feature_maps_cn, feature_maps_UN)))
        avg_name.append(method)
        n.append(layer_name)
        
df_acti = pd.DataFrame({'jensen-shannon':np.asarray(acti), 'method':category, 'img_name':img_list})
df_avg_jen = pd.DataFrame({'average_jensen_shannon':np.asarray(avg_jen), 'method':avg_name, 'layer_name':n})
df_acti.to_csv(save_path + "jensen_shannon_"  + layer_name + '.csv')
df_avg_jen.to_csv(save_path + "jensen_shannon_avg_" + layer_name  + '.csv')


#plt.figure()
#sns.boxplot(x = 'method',y = 'jensen-shannon', data = df_acti)
#plt.title('activation_25')

# %%

df1 = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/jensen_shannon_avg__conv2d.csv", index_col = 0)
df2 = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/jensen_shannon_avg__activation_1.csv", index_col = 0)
df3 =  pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/jensen_shannon_avg_decoder0_conv2d_20_UN.csv", index_col = 0)
df4 =  pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/jensen_shannon_avg_decoder0_activation_25.csv", index_col = 0)

df_jen = pd.read_excel("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/df_jen.xlsx", index_col =0)
# %%

plt.figure()
df_jen.plot()
plt.title('Average Jensen-Shannon Divergence')
# %%

#for layer in model_Conv2D.layers:
#	# check for convolutional layer
#	if 'conv' not in layer.name:
#		continue
#	
#	# get filter weights
#	filters, biases = layer.get_weights()
#	print(layer.name, filters.shape)
#
#filters, biases = model_Conv2D.layers[1].get_weights()
#
## normalize filter values to 0-1 so we can visualize them
#f_min, f_max = filters.min(), filters.max()
#filters = (filters - f_min) / (f_max - f_min)
# 
##    plot first few filters
#n_filters, ix = 6, 1
#for i in range(n_filters):
#	# get the filter
#	f = filters[:, :, :, i]
#	# plot each channel separately
#	for j in range(3):
#		# specify subplot and turn of axis
#		ax = plt.subplot(n_filters, 3, ix)
#		ax.set_xticks([])
#		ax.set_yticks([])
#		# plot filter channel in grayscale
#		plt.imshow(f[:, :, j], cmap='gray')
#		ix += 1
## show the figure
#plt.show()

# %%
<<<<<<< Updated upstream
model_gray = models.load_model('F:/CP8309/Project/models/rgb/modelsweights_2019_12_5_12_42_42_weights.37--0.98.hdf5', custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
=======
model_gray = models.load_model('E:/Pathcore/Results_v2/Models/UNET3_gen_models/chosen_models/weights_2019_8_23_17_11_42_UN_weights.40-1.02.hdf5', custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
>>>>>>> Stashed changes
# summarize feature map shapes
for i in range(len(model_gray.layers)):
    layer = model_gray.layers[i]
    # check for convolutional layer
    if 'activation' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)
    
model_UN = models.load_model('F:/CP8309/Project/models/rgb/modelsweights_2019_12_5_12_42_42_weights.37--0.98.hdf5', custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
# summarize feature map shapes
for i in range(len(model_UN.layers)):
	layer_UN = model_UN.layers[i]
	# check for convolutional layer
	if 'activation' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)
	
# %%
# redefine model to output right after the first hidden layer

model_GRAY = models.Model(inputs=model_UN.inputs, outputs = model_UN.layers[6].output)
model_un = models.Model(inputs=model_UN.inputs, outputs = model_UN.layers[6].output)    

# %%
ix = 1
<<<<<<< Updated upstream
plt.figure()
for _ in range(8):
    for _ in range(4):
        
        ax = pyplot.subplot(8,4,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps_RH[0, :, :, ix-1], cmap='jet')
#        pyplot.hist(feature_maps[0, :, :, ix-1].ravel(), 75)
        ix += 1
		
# %%
        
df_gan = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/GAN_activation.csv", index_col = 0, names = ['GAN'])
df_HS = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/HS_activation.csv", index_col = 0, names = ['HS'])
df_KH = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/KH_activation.csv", index_col = 0, names = ['KH'])
df_MC = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/MC_activation.csv", index_col = 0, names = ['MC'])
df_RH = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/RH_activation.csv", index_col = 0, names = ['RH'])
df_gray = pd.read_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/gray_activation.csv", index_col = 0, names = ['Gray'])

df = pd.concat([df_gan,df_HS,df_KH,df_MC,df_RH,df_gray], axis = 1)

# %%
plt.savefig('D:/gitRepo/justinpontalba/CP8309_Project/figures/Jensen_shannon_all.png')
=======


plt.figure()
for _ in range(8):
	for _ in range(4):
		
		ax = pyplot.subplot(8,4,ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(feature_maps_RH[0, :, :, ix-1], cmap='jet')
#		pyplot.hist(feature_maps[0, :, :, ix-1].ravel(), 75)
		ix += 1

>>>>>>> Stashed changes

# %%
gan_count = 0
hs_count = 0
mc_count = 0
kh_count = 0
rh_count = 0
gray_count = 0

for gan,hs,kh,mc,rh, gray in zip(df['GAN'], df['HS'], df['KH'], df['MC'], df['RH'], df['Gray']):
    
    if gan > 0.5:
        gan_count = gan_count + 1
    if hs > 0.5:
        hs_count = hs_count + 1
    if kh > 0.5:
        kh_count = kh_count + 1
    if mc > 0.5:
        mc_count = mc_count + 1
    if rh > 0.5:
        rh_count = rh_count + 1
    if gray > 0.5:
        gray_count = gray_count + 1
        
df_count = pd.DataFrame({'GAN':gan_count,'HS':hs_count,'KH':kh_count,'MC':mc_count,'RH':rh_count,'Gray':gray_count}, index = ['Count'])
df_count = df_count.transpose()
df_count.to_csv("G:/My Drive/Master's Courses/CP8309 - Neural Information Processing/Project/Results/shannon_jenson_count.csv")

# %%
plt.figure()
sns.barplot(x = df_count.index, y = df_count['Count'])
plt.title('Un-Normalized model: Counts of Filters Above Threshold For Jensen-Shannon Divergence')

# %%
plt.savefig('D:/gitRepo/justinpontalba/CP8309_Project/figures/Jensen_shannon_all_counts.png')

# %%


