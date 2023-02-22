# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:51:36 2019

@author: Justi
"""
#---------------------Libraries--------------------#
import numpy as np
import os
from skimage import io,filters, morphology
import pandas as pd
from sklearn.preprocessing import minmax_scale
from skimage.filters import threshold_otsu
#from utils import tensorwrapper, dice_coeff, dice_loss, bce_dice_loss
import scipy.stats

#---------------------Tensorflow Libraries--------------------#
from tensorflow.python.keras import models

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

def postProcessBoundary(boundary):
    
#    med_filt = filters.median(boundary*255)
    boundary = boundary.astype(bool)
    rso_img = morphology.remove_small_objects(boundary, min_size = 20 )
#    plt.figure()
#    plt.imshow(rso_img)
    try:
        rso_img = rso_img.astype(int)
    except ValueError:
        print('conversion error')
    
    return rso_img

def postProcessNuclei(nuclei):
    
    
    med_filt = filters.median(nuclei)
    close_img = morphology.binary_closing(med_filt)
    rso_img = morphology.remove_small_objects(close_img, min_size = 30 )
    try:
        rso_img = rso_img.astype(int)
    except ValueError:
        print('conversion error')
    
    return rso_img


def imBinarize(img):
    
    n = img[:,:,0]
    bo = img[:,:,1]
    ba = img[:,:,2]
    
    thresh1 = threshold_otsu(n)
    thresh2 = threshold_otsu(bo)
    thresh3 = threshold_otsu(ba)
    
    nuclei = n > thresh1
    boundary = bo > thresh2
    background = ba > thresh3
    
    
#    img = np.stack((nuclei,boundary,background), axis = 2)
#    
#    max_label = np.argmax(img, axis = 2)
#    nuclei = (max_label == 0).astype(int)
#    boundary = (max_label == 1).astype(int)
#    background = (max_label == 2).astype(int)
    
    return nuclei.astype(int), boundary.astype(int), background.astype(int)

def getValidationMetrics(test_image_path, gt_path,pred_save_path,mask_save_path, model_path,name):
    """ This  function will return validation metrics
    
    Args:
        test_image_path (str): Path to test images
        gt_path (str): Path to ground truths
        save_path (str): Path to saving entire prediction
        mask_save_path (str): Path to saving masks
        model_path (str): Path to model
        name (str): experiment name
    
    
    Returns:
        df (dataframe): dataframe of validation metrics
        
    """
    
    print('Reading images from:', test_image_path)
    print('Reading labels from:', gt_path)
    print('Saving predictions to:',pred_save_path)
    print('Saving masks to:', mask_save_path)
    print('Reading model:', model_path)
    print('Name of experiment:', name)
    
    model = models.load_model(model_path, custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})
    
    
    dsc_list = []
    ef_list = []
    prec_list = []
    rec_list = []
    categ_list = []
    
    test_image_list = os.listdir(test_image_path)
    test_image_list = pd.DataFrame(test_image_list)
    test_image_list = test_image_list.sort_values(by = 0)
    
    gt_list = os.listdir(gt_path)
    gt_list = pd.DataFrame(gt_list)
    gt_list = gt_list.sort_values(by = 0)
    
    for i,gt in zip(test_image_list[0], gt_list[0]):
        
        print('predicting:',i,gt)
        img = io.imread(test_image_path + i)
        gt_img = io.imread(gt_path + gt)
        img = tensorwrapper(img)
        
        pred = model.predict(img/255)
        pred = np.squeeze(pred)
        
        nuclei, boundary, background =  imBinarize(pred/255)
        nuclei_gt, boundary_gt, background_gt =  imBinarize(gt_img/255)
        
        nuclei = postProcessNuclei(nuclei)
        boundary = postProcessBoundary(boundary)
        
        ef,dsc,tp,tn,fp,fn, precision, recall = validation(nuclei,nuclei_gt)
        
        dsc_list.append(dsc)
        ef_list.append(ef)
        prec_list.append(precision)
        rec_list.append(recall)
        categ_list.append(name)
        
        io.imsave(pred_save_path + i, pred)
        io.imsave(mask_save_path + i, nuclei)
         
    df = pd.DataFrame({'dsc':dsc_list, 'ef':ef_list, 'pr':prec_list, 'rc':rec_list, 'type':categ_list})
    
    return df


    
    
def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance
    
    
def activationMetrics(unit_a, unit_b):
    
    a = np.squeeze(unit_a)
    b = np.squeeze(unit_b)
    
    jen_sen = []
    shape = np.shape(a)
    for i in range(shape[2]):
        
        a_scaled = minmax_scale(a[:,:,i], feature_range = (0,1))
        b_scaled = minmax_scale(b[:,:,i], feature_range = (0,1))
        jen_sen.append(jensen_shannon_distance(a_scaled.ravel(), b_scaled.ravel()))
        
    return jen_sen
        
        
    
    
    