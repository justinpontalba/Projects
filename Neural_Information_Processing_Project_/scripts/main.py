# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:52:27 2019

@author: Justi
"""
# %%
#---------------------Libraries--------------------#
import os
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)
import pandas as pd
import click

#---------------------Classes--------------------#
from unet import Unet
from loadData import dataLoader

# %%

@click.command()
@click.option("--x_train_path", default='D:/gitRepo/justinpontalba/CP8309_Project/data/train/UN/images/')
@click.option("--y_train_path", default='D:/gitRepo/justinpontalba/CP8309_Project/data/train/UN/labels/')
@click.option("--x_val_path", default='D:/gitRepo/justinpontalba/CP8309_Project/data/val/UN/images/')
@click.option("--y_val_path", default='D:/gitRepo/justinpontalba/CP8309_Project/data/val/UN/labels/')
@click.option("--log_path", default='D:/gitRepo/justinpontalba/CP8309_Project/logs/')
@click.option("--base_path", default= 'D:/CP8309/Project/models/')
@click.option("--batch_size", default = 3)
@click.option("--epochs", default = 40)
@click.option("--img_dim", default = 256)
@click.option("--n_channels", default = 3)

# %%

def main(x_train_path,y_train_path,x_val_path,y_val_path,log_path,base_path, batch_size, epochs,img_dim, n_channels):
    
    print('Setting Training Parameters...')
    print('Training Paths:...', x_train_path, y_train_path)
    print('Validation Paths:', x_val_path, y_val_path)
    print('Tensorboard Log:', log_path)
    print('Model Save Path:', base_path)
    print('Batch size:', batch_size)
    print('Epochs:', epochs)
    print('Input dimension:', img_dim)
    print('N-Channels:', n_channels)
    
    img_shape = (img_dim,img_dim,n_channels)
    
    x_train_filenames = pd.DataFrame(os.listdir(x_train_path))
    x_train_filenames = x_train_filenames.sort_values(by = 0)
    x_train_filenames = x_train_filenames[0]
    
    y_train_filenames = pd.DataFrame(os.listdir(y_train_path))
    y_train_filenames = y_train_filenames.sort_values(by = 0)
    y_train_filenames = y_train_filenames[0]
    
    x_val_filenames = pd.DataFrame(os.listdir(x_val_path))
    x_val_filenames = x_val_filenames.sort_values(by = 0)
    x_val_filenames = x_val_filenames[0]
    
    y_val_filenames = pd.DataFrame(os.listdir(y_val_path))
    y_val_filenames = y_val_filenames.sort_values(by = 0)
    y_val_filenames = y_val_filenames[0]
    
    for i in range(len(x_train_filenames)):
        
        temp = x_train_filenames[i]
        x_train_filenames[i] = x_train_path + temp
        y_train_filenames[i] = y_train_path + temp
        
    for j in range(len(x_val_filenames)):
        temp2 = x_val_filenames[j]
        x_val_filenames[j] = x_val_path + temp2
        y_val_filenames[j] = y_val_path + temp2
    

    
    num_train_examples = len(x_train_filenames)
    num_val_examples = len(x_val_filenames)
    
    data = dataLoader(img_shape, x_train_filenames, y_train_filenames, x_val_filenames, y_val_filenames, 3)
    
    train_ds, val_ds = data.setParams()
    
    unet_model = Unet(batch_size, epochs, 'Adam', num_train_examples, num_val_examples, log_path, base_path)
    
    unet_model.train(train_ds,val_ds)

if __name__ == "__main__":
    main()
