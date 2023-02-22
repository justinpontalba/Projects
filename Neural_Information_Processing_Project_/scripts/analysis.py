# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:25:24 2019

@author: jpontalb
"""

# %%

from utils import *
import numpy as np
import pandas as pd
import os
from skimage import measure, color
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras import models 
from skimage.filters import threshold_otsu
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from unet import *

# %%

im_ref = open_image('D:/gitRepo/justinpontalba/nuseg/Colour Normalization/For the Paper/stain_normalisation_toolbox/ref1.png')
ref_lab = color.rgb2lab(im_ref,illuminant = "D65", observer = "2")
ref_a = ref_lab[:,:,1]
ref_b = ref_lab[:,:,2]



method_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/'
method_dir = os.listdir(method_path)

chosen_path = 'F:/unet_chosen_models/'
chosen_dir = os.listdir(chosen_path)

gt_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/UN/labels/'
gt_dir = os.listdir(gt_path)

# %%
for meth,mod in zip(method_dir, chosen_dir):
    
    alpha = []
    beta = []
    nmh_metric = []
    nmi_metric = []
    dsc = []
    print('Starting...')
    
    model = models.load_model(chosen_path + mod,custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
    
    im_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/' + meth + '/images/'
    im_dir = os.listdir(im_path)
    
    for i,gt in zip(im_dir, gt_dir):
        
        print(i,gt)
        read_path = im_path + i
        read_gt = gt_path + gt
        print(read_path)
        
        im_gt = open_image(gt_path + gt)
        im_tar = open_image(read_path)
        
        tar_lab = color.rgb2lab(im_tar,illuminant = "D65", observer = "2")
        
        tar_a = tar_lab[:,:,1]
        tar_b = tar_lab[:,:,2]
        
        amce_a = amce_lab(ref_a,tar_a,25)
        amce_b = amce_lab(ref_b,tar_b,25)
        nmi = getNMI(im_tar)
        nmh = getNMH(im_tar)
        
        alpha.append(amce_a)
        beta.append(amce_b)
        nmh_metric.append(nmh)
        nmi_metric.append(nmi)
        
        im_predict, im_test, img_gt = testModel(read_path, model, read_gt)
        gt_nuclei = img_gt[:,:,0]/255
    
        
        nuclei, boundary, background = imBinarize(im_predict)
        nuclei_post = postProcessNuclei(nuclei)
        
        EF,DSC,TP,TN,FP,FN, precision, recall = validation(nuclei_post,gt_nuclei)
        gt_nuclei = minmax_scale(gt_nuclei)
        
        dsc.append(DSC)
    
    print('Ending...')    
    df = pd.DataFrame({'alpha_a':alpha, 'alpha_b': beta, 'NMI':nmi_metric, 'NMH':nmh_metric, 'DSC':dsc})
    df.to_csv('D:/gitRepo/justinpontalba/nuseg/investigating further/' + meth + '_metrics.csv')

# %%
# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
lin = PolynomialFeatures(degree=2)


# %%
df_un = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/UN_metrics.csv', index_col = 0) 
df_rh = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/RH_metrics.csv', index_col = 0) 
df_mc = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/MC_metrics.csv', index_col = 0) 
df_kh = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/KH_metrics.csv', index_col = 0 )
df_hs = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/KH_metrics.csv', index_col = 0 )
df_gan = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/GAN_metrics.csv', index_col = 0)


x = df_un[{'alpha_a','alpha_b', 'NMI', 'NMH'}]
y = df_un['DSC']

x_rh = df_rh[{'alpha_a','alpha_b', 'NMI', 'NMH'}]
y_rh = df_rh['DSC']

df_loc = os.listdir('E:/IAMGit/justinpontalba/nuseg/investigating further/')

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)
ind = x_train.index
ind = np.asarray(ind)

ind_test = x_test.index
ind_test = np.asarray(ind_test)

mse = []
name = []


for file in df_loc:
    
    df = pd.read_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/' + file, index_col =0)
    x = df[{'alpha_a','alpha_b', 'NMI', 'NMH'}]
    y = df['DSC']
    
    x_train = x.loc[ind]
    x_test = x.loc[ind_test]
    
    y_train = y.loc[ind]
    y_test = y.loc[ind_test]
    
    lin.fit(x_train, y_train)
    y_predict = lin.predict(x_test)
    
    n = file.split('_')
    n = n[0]
    mse.append(mean_squared_error(y_test,y_predict))
    name.append(n)
    plt.figure()
    plt.scatter(y_test,y_predict)
    plt.scatter(y_test,y_test)
    plt.title(n)


 

# %%

df_mse =pd.DataFrame({'mse':mse}, index = name)
df_mse.to_csv('E:/IAMGit/justinpontalba/nuseg/investigating further/mse_TCGA.csv')

plt.figure()
plt.scatter(y_predict, y_test, color = 'red')
plt.scatter(y_test,y_test, color = 'blue')

