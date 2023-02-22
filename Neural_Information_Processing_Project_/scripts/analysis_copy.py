# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:22:45 2019

@author: jpontalb
"""

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
from validation_metrics_results import imBinarize,postProcessNuclei

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import minmax_scale
from sklearn import linear_model
from scipy.linalg import eig

# %%

im_ref = open_image('D:/gitRepo/justinpontalba/nuseg/Colour Normalization/For the Paper/ref1.png')
ref_lab = color.rgb2lab(im_ref,illuminant = "D65", observer = "2")
ref_a = ref_lab[:,:,1]
ref_b = ref_lab[:,:,2]

method_path = 'F:/Paper Data/UNET3/Datasets/SMH/images/'
method_dir = os.listdir(method_path)

chosen_path = 'F:/unet_chosen_models/'
chosen_dir = os.listdir(chosen_path)

gt_path = 'F:/Paper Data/UNET3/Datasets/SMH/nuclei/sub_patches/'
gt_dir = os.listdir(gt_path)
#gt_dir = gt_dir[0:49]

# %%

alpha = []
beta = []
nmh_metric = []
nmi_metric = []
dsc = []
data_set = []
method = []

dname = 'SMH'

for meth,mod in zip(method_dir, chosen_dir):
    

    print('Starting...')
    
    model = models.load_model(chosen_path + mod,custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
    
    im_path = method_path + meth + '/sub_patches/'
    im_dir = os.listdir(im_path)
#    img_dir = im_dir[0:49]
    
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
        method.append(meth)
        data_set.append(dname)
        
        
        im_predict, im_test, img_gt = testModel(read_path, model, read_gt)
        gt_nuclei = img_gt/255
    
        
        nuclei, boundary, background = imBinarize(im_predict)
        nuclei_post = postProcessNuclei(nuclei)
        
        EF,DSC,TP,TN,FP,FN, precision, recall = validation(nuclei_post,gt_nuclei)
        gt_nuclei = minmax_scale(gt_nuclei)
        
        dsc.append(DSC)
    
print('Ending...')    

# %%
df = pd.DataFrame({'alpha_a':alpha, 'alpha_b': beta, 'NMI':nmi_metric, 'NMH':nmh_metric, 'DSC':dsc, 'method':method, 'source':data_set})
df.to_csv('D:/gitRepo/justinpontalba/nuseg/investigating further/all_methods_metrics.csv')


# %%

plt.figure()
sns.boxplot(x = 'source', y = 'DSC', data = df, hue = 'method', hue_order= ['UN','GAN', 'HS', 'KH', 'MC','RH'])


# %%
dname = 'TNBC'
df_path = 'D:/gitRepo/justinpontalba/nuseg/investigating further/'+ dname + '/all_methods_metrics.csv'
save_path = 'D:/gitRepo/justinpontalba/nuseg/investigating further/' + dname + '/'


df = pd.read_csv(df_path, index_col = 0)
unique = np.unique(df['method'])
mse = []
name = []
dataset = []

poly = PolynomialFeatures(degree=3)


for file,i in zip(unique, range(len(unique))):
    
    print(file,i)
    
    df = pd.read_csv(df_path, index_col = 0)
    df = df[df['method']==file]
    
    df['alpha_a'] = minmax_scale(df['alpha_a'])
    df['alpha_b'] = minmax_scale(df['alpha_b'])
    df['NMI'] = minmax_scale(df['NMI'])
    df['NMH'] = minmax_scale(df['NMH'])
    
    x = df[{'alpha_a','alpha_b', 'NMI', 'NMH'}]
    y = df['DSC']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)
    

    
    x_poly = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly,y_train)
    
    x_test_poly = poly.fit_transform(x_test)
    
    y_poly_pred = model.predict(x_test_poly)
    
    

    mse.append(mean_squared_error(y_test,y_poly_pred))
    name.append(file)
    dataset.append(dname)
    
    plt.subplot(3,2,i + 1)
    plt.scatter(y_test,y_poly_pred)
    plt.plot(y_test,y_test, color = 'red')
    plt.ylabel('y-predict')
    plt.title(file)
plt.suptitle('Polynomial Regression: y_true vs y_prediction')

# %%
df_mse =pd.DataFrame({'mse':mse, 'source':dataset}, index = name)
df_mse.to_csv(save_path + 'mse.csv')

# %%
mse_tcga = pd.read_csv('D:/gitRepo/justinpontalba/nuseg/investigating further/TCGA/mse.csv')
mse_tnbc = pd.read_csv('D:/gitRepo/justinpontalba/nuseg/investigating further/TNBC/mse.csv')
mse_smh = pd.read_csv('D:/gitRepo/justinpontalba/nuseg/investigating further/SMH/mse.csv')

#mse_tcga = mse_tcga.filter({'mse'})
#mse_tnbc = mse_tnbc.filter({'mse'})
#mse_smh = mse_smh.filter({'mse'})

df_mse = pd.concat([mse_tcga,mse_smh, mse_tnbc])
df_mse = df_mse.rename(columns = {'Unnamed: 0':'method', 'mse':'mse', 'source':'source'})

plt.figure()
sns.barplot(x = 'source', y = 'mse', data = df_mse, hue = 'method', hue_order= ['UN','GAN', 'HS', 'KH', 'MC','RH'])
plt.ylim(0, 1)
plt.title('Mean-squared Error')

# %%
from sklearn.decomposition import PCA

dname = 'TCGA'
df_path = 'D:/gitRepo/justinpontalba/nuseg/investigating further/'+ dname + '/all_methods_metrics.csv'
save_path = 'D:/gitRepo/justinpontalba/nuseg/investigating further/' + dname + '/'
df = pd.read_csv(df_path, index_col = 0)
unique = np.unique(df['method'])

for file,i in zip(unique, range(len(unique))):
    
    print(file,i)
    df = pd.read_csv(df_path, index_col = 0)
    df = df[df['method']==file]
    df['alpha_a'] = minmax_scale(df['alpha_a'])
    df['alpha_b'] = minmax_scale(df['alpha_b'])
    df['NMI'] = minmax_scale(df['NMI'])
    df['NMH'] = minmax_scale(df['NMH'])
    
    x = df[{'alpha_a','alpha_b', 'NMI', 'NMH'}]
    y = df['DSC']
    
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)
    
    
    x_poly = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly,y_train)
    
    x_test_poly = poly.fit_transform(x_test)
    y_poly_pred = model.predict(x_test_poly)
    
    mse.append(mean_squared_error(y_test,y_poly_pred))
    name.append(file)
    dataset.append(dname)
    
    plt.subplot(3,2,i + 1)
    plt.subplot(3,2,i + 1)
    plt.scatter(y_test,y_poly_pred)
    plt.plot(y_test,y_test, color = 'red')
    plt.ylabel('y-predict')
    plt.title(file)
    
plt.suptitle('Polynomial Regression: y_true vs y_prediction')