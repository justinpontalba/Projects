# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:52:26 2019

@author: Justi
"""

# %%

#---------------------Libraries--------------------#
import datetime
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)
from skimage import io
from skimage.util import view_as_windows
from skimage.color import rgb2hsv
import cv2
from tensorflow.python.keras import losses

#---------------------Tensorflow Libraries--------------------#
from tensorflow.python.keras import models
import tensorflow as tf

# %%

def tensorwrapper(arr):
    
    arr_w = np.expand_dims(arr, axis = 0)
    
    return arr_w

def getSavePath(base_path):
    
    currentDT = datetime.datetime.now()
    weights_name = 'weights_' + str(currentDT.year) + '_' + str(currentDT.month) + '_' + str(currentDT.day) + '_' + str(currentDT.hour) + '_' + str(currentDT.minute) + '_' + str(currentDT.second)
    save_model_path =  base_path +  weights_name + '_' + 'weights.{epoch:02d}--{val_loss:.2f}.hdf5'
    
    return save_model_path


def intermediateLayer(model, layer_name):
    
    print('Target Layer:', layer_name)
    inputs = model.input
    intermediate_layer_model = models.Model(inputs = inputs, outputs = model.get_layer(layer_name).output)
    
    return intermediate_layer_model
    
    
def open_image(path):
    """
    Open an image from disk as a numpy array.
    
    Parameters
    ----------
    path : string
        path of file to open
    
    Returns
    -------
    numpy.ndarray
    """
    return io.imread(path)


# Save an image
def save_image(path, data):
    """
    Save a numpy array to disk as an image

    Parameters
    ---------
    path : string
        path of file to save
    data: numpy.ndarray
        data to save

    Returns
    -------
    None
    """
    io.imsave(path, data)

def amce_lab(tar,proc,w):
    

    
    window_shape = (w,w)
    tar_w = view_as_windows(tar,window_shape,10)
    proc_w = view_as_windows(proc,window_shape,10)
    
    tar_w = tar_w.reshape(np.shape(tar_w)[0]*np.shape(tar_w)[1], window_shape[0]*window_shape[1])
    proc_w = proc_w.reshape(np.shape(proc_w)[0]*np.shape(proc_w)[1], window_shape[0]*window_shape[1])
    
    W_tar = len(tar_w)
    W_proc = len(proc_w)
    
    mu_tar = np.mean(tar_w,axis = 1).reshape(-1,1)
    mu_pro = np.mean(proc_w, axis = 1).reshape(-1,1)
    
    amce = np.abs((1/W_tar) * np.sum(mu_tar) - (1/W_proc) * np.sum(mu_pro))
    
    return amce

def dice_coeff( y_true, y_pred):
    
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def overlapStats(A,B):
    
    A[A>0] = 1
    
    B[B>0] = 2
    
    overlay = A + B
    
    TP = np.size(np.where(overlay == 3))
    TN = np.size(np.where(overlay == 0))
    FP = np.size(np.where(overlay == 2))
    FN = np.size(np.where(overlay == 1))
    
    return TP, TN, FP, FN;

def validation(A,B):
    # Add both the parameters and return them."
    [TP, TN, FP, FN] = overlapStats(A,B)
    
    DSC = (2*TP)/(2*TP + FP + FN)
    
    try:
        EF = FP / (TP + FN)
    except ZeroDivisionError:
        
        EF = 'Error'
           
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 'Error'
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 'Error'
    
    return EF,DSC,TP,TN,FP,FN, precision, recall

def getNMI(img):
    
    r = img[:,:,0].ravel()
    g = img[:,:,1].ravel()
    b = img[:,:,2].ravel()
    
    rgb_combined = np.array([r,g,b])
    rgb_combined = rgb_combined.T
    mean_triplet = np.mean(rgb_combined, axis = 1)
    med = np.median(mean_triplet)
    per = np.percentile(mean_triplet,95, axis = 0)
    
    nmi = (med/per).astype(float)
    
    return nmi

def getNMH(img):
    
    i_hsv = rgb2hsv(img)
    iH,iS,iV = cv2.split(i_hsv)
    
    iH = iH.ravel()
    med = np.median(iH)
    per = np.percentile(iH, 95, axis = 0)
    nmh = (med/per).astype(float)
    
    return nmh

def testModel(test_path,model, gt_path):
    
    img_test = open_image(test_path)
    img_gt = open_image(gt_path)
    
    img_test= img_test/255
    img_tense = tensorwrapper(img_test)  
    predict = model.predict(img_tense)
    predict = np.squeeze(predict)
    
    return predict, img_test, img_gt
