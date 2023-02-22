# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:58:59 2019

@author: Justi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import io
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models 
from sklearn.model_selection import train_test_split


base = 'E:/Pathcore/Results_v2/Architectures/UNET3/Datasets/TCGA-Kumar/images/'

meth_list = os.listdir(base)

for meth in meth_list:

	img_path = base + meth + '/images/'
	img_dir = os.listdir(img_path)
	gt_path = 'E:/Pathcore/Results_v2/Architectures/UNET3/Datasets/TCGA-Kumar/ternary_GT/'
	gt_dir = os.listdir(gt_path)
	
	write_dir_path_train = 'E:/IAMGit/justinpontalba/CP8309_Project/data/train/' + meth + '/'
	write_dir_path_test = 'E:/IAMGit/justinpontalba/CP8309_Project/data/test/' + meth + '/'
	write_dir_path_val = 'E:/IAMGit/justinpontalba/CP8309_Project/data/val/' + meth + '/'
	
	try:
		os.mkdir(write_dir_path_train)
		
	except FileExistsError:
		print('Folder Exists')
		
	try:
		os.mkdir(write_dir_path_test)
	except FileExistsError:
		print('Folder Exists')
	try:
		os.mkdir(write_dir_path_val)
	except FileExistsError:
		print('Folder Exists')	
	
	new_img_dir_train = write_dir_path_train + '/images/'
	new_gt_dir_train =  write_dir_path_train + '/labels/'
	new_img_dir_val = write_dir_path_val + '/images/'
	new_gt_dir_val =  write_dir_path_val + '/labels/'
	new_img_dir_test = write_dir_path_test + 'images/'
	new_gt_dir_test =  write_dir_path_test + 'labels/'
	
	manifest = pd.read_csv(base + meth + '/manifest_files/new_manifest.csv', index_col = 0)
	test_manifest = manifest[manifest['type'] == 'test']
	train_manifest = manifest[manifest['type'] == 'train']
	train_manifest_X = train_manifest['img_path']
	train_manifest_Y = train_manifest['gt_path']
	
	x_train,x_val, y_train, y_val = train_test_split(train_manifest_X, train_manifest_Y, test_size = 0.17, random_state = 0 )
	
	try:
	    os.mkdir(new_img_dir_train)
	except FileExistsError:
	    print('Folder Exists')
	try:
	    os.mkdir(new_gt_dir_train)
	except FileExistsError:
	    print('Folder Exists')
	
	try:
	    os.mkdir(new_img_dir_test)
	except FileExistsError:
	    print('Folder Exists')
	try:
	    os.mkdir(new_gt_dir_test)
	except FileExistsError:
	    print('Folder Exists')
	
	try:
	    os.mkdir(new_img_dir_val)
	except FileExistsError:
	    print('Folder Exists')
	try:
	    os.mkdir(new_gt_dir_val)
	except FileExistsError:
	    print('Folder Exists')
	
	for item, gt in zip(img_dir,gt_dir):
	    print('Checking agreement...')
	    for item2 in x_train:
	        
	        name = item2.split('/')
	        img_name = name[9]
	#        print(item + '------------------' + img_name, status)
	        if item == img_name:
	#            print('We got a match')
	            i = io.imread(img_path + item)
	            g = io.imread(gt_path + gt)
	            new_save_gt = gt.split('m_')
	#            print(new_img_dir_train + item,new_gt_dir_train + new_save_gt[1])
	            io.imsave(new_img_dir_train + item, i)
	            io.imsave(new_gt_dir_train + new_save_gt[1], g)
	
	    for item3 in test_manifest['img_path']:
	        
	        name = item3.split('/')
	        img_name = name[9]
	#        print(item + '------------------' + img_name, status)
	        if item == img_name:
	#            print('We got a match')
	            i = io.imread(img_path + item)
	            g = io.imread(gt_path + gt)
	            new_save_gt = gt.split('m_')
	#            print(new_img_dir_train + item,new_gt_dir_train + new_save_gt[1])
	            io.imsave(new_img_dir_test + item, i)
	            io.imsave(new_gt_dir_test + new_save_gt[1], g)
	            
	    for item4 in x_val:
	        
	        name = item4.split('/')
	        img_name = name[9]
	#        print(item + '------------------' + img_name, status)
	        if item == img_name:
	#            print('We got a match')
	            i = io.imread(img_path + item)
	            g = io.imread(gt_path + gt)
	            new_save_gt = gt.split('m_')
	#            print(new_img_dir_train + item,new_gt_dir_train + new_save_gt[1])
	            io.imsave(new_img_dir_val + item, i)
	            io.imsave(new_gt_dir_val + new_save_gt[1], g)

            
            
            
            
            
        
        
        
    
    
    
    
    
    
    
    
    
    





