# -*- coding: utf-8 -*-

# Import necessary libraries.
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from utils import *
from unet import *
from validation_metrics_results import *
from skimage import io, color 
from numpy import expand_dims

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from tempfile import TemporaryFile
outfile = TemporaryFile()
import os
import random
import pandas as pd


# %%

model = models.load_model('F:/CP8309/Project/models/chosen_models/UN_weights_2019_12_5_12_42_42_weights.37--0.98.hdf5', 
                          custom_objects = {'bce_dice_loss':bce_dice_loss, 'dice_loss':dice_loss, 'dice_coeff':dice_coeff})

# %%
#---------------------------Set up---------------------------#

x_test_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/GAN/images/'
y_test_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/test/GAN/labels/'
save_loc = 'F:/CP8309/Project/reconstruction/test/prediction_conv2d_20_GAN/'
raw_save = 'F:/CP8309/Project/reconstruction/test/raw_conv2d_20_GAN/'

try:
    os.mkdir(save_loc)
except FileExistsError:
    print('Folder already exists')
try:
    os.mkdir(raw_save)
except FileExistsError:
    print('Folder already exists')

x_test_filenames = pd.DataFrame(os.listdir(x_test_path))
x_test_filenames = x_test_filenames.sort_values(by = 0)
x_test_filenames = x_test_filenames[0]

y_test_filenames = pd.DataFrame(os.listdir(y_test_path))
y_test_filenames = y_test_filenames.sort_values(by = 0)
y_test_filenames = y_test_filenames[0]

for i in range(len(x_test_filenames)):
    
    temp = x_test_filenames[i]
    x_test_filenames[i] = x_test_path + temp
    y_test_filenames[i] = y_test_path + temp

x_train_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/train/GAN/images/'
y_train_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/train/GAN/labels/'
x_val_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/val/GAN/images/'
y_val_path = 'D:/gitRepo/justinpontalba/CP8309_Project/data/val/GAN/labels/'

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


# %%
for layer in model.layers:
    print(layer.name)
    
# %%

layer_name = 'conv2d_20'
inputs = model.input
intermediate_layer_model = models.Model(inputs = inputs, outputs = model.get_layer(layer_name).output)
image_shape=(1,256,256,3)


# %%
for x in range(0,len(x_test_filenames)):
    print(x)
    

#    rand_int = random.randint(0,len(x_test_filenames))
    x_image = io.imread(x_test_filenames[x])
    x_image = expand_dims(x_image, axis = 0)
    x_image = x_image/255
      
    intermediate_layer_output = intermediate_layer_model.predict(x_image)
    X0 = intermediate_layer_output


    X=tf.Variable(tf.zeros(image_shape))
    # Encoder0 encoder0_pool, encoder0 = encoder_block(conv1, 32) # 128
    encoder0_conv2d = layers.Conv2D(32, (3, 3), padding='same')(X)
    encoder0 = layers.BatchNormalization()(encoder0_conv2d)    
    encoder0_activation = layers.Activation('relu')(encoder0)
    encoder0_conv2d_1 = layers.Conv2D(32, (3, 3), padding='same')(encoder0_activation)
    encoder0 = layers.BatchNormalization()(encoder0_conv2d_1)
    encoder0_activation1 = layers.Activation('relu')(encoder0)
    encoder_pool0 = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder0_activation1)
    
    # Encoder1 encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    encoder1 = layers.Conv2D(64, (3, 3), padding='same')(encoder_pool0)
    encoder1 = layers.BatchNormalization()(encoder1)    
    encoder1 = layers.Activation('relu')(encoder1)
    encoder1 = layers.Conv2D(64, (3, 3), padding='same')(encoder1)
    encoder1 = layers.BatchNormalization()(encoder1)
    encoder1 = layers.Activation('relu')(encoder1)
    encoder_pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder1)
    
    # Encoder2 encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    encoder2 = layers.Conv2D(128, (3, 3), padding='same')(encoder_pool1)
    encoder2 = layers.BatchNormalization()(encoder2)    
    encoder2 = layers.Activation('relu')(encoder2)
    encoder2 = layers.Conv2D(128, (3, 3), padding='same')(encoder2)
    encoder2 = layers.BatchNormalization()(encoder2)
    encoder2_activation5 = layers.Activation('relu')(encoder2)
    encoder_pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder2_activation5)
    
    # Encoder3 encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    encoder3 = layers.Conv2D(256, (3, 3), padding='same')(encoder_pool2)
    encoder3 = layers.BatchNormalization()(encoder3)    
    encoder3 = layers.Activation('relu')(encoder3)
    encoder3 = layers.Conv2D(256, (3, 3), padding='same')(encoder3)
    encoder3 = layers.BatchNormalization()(encoder3)
    encoder3 = layers.Activation('relu')(encoder3)
    encoder_pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder3)
    
    # Encoder4 encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
    encoder4 = layers.Conv2D(512, (3, 3), padding='same')(encoder_pool3)
    encoder4 = layers.BatchNormalization()(encoder4)    
    encoder4 = layers.Activation('relu')(encoder4)
    encoder4 = layers.Conv2D(512, (3, 3), padding='same')(encoder4)
    encoder4 = layers.BatchNormalization()(encoder4)
    encoder4 = layers.Activation('relu')(encoder4)
    encoder_pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder4)
    
    # Center center = conv_block(encoder4_pool, 1024) # center
    center = layers.Conv2D(1024, (3, 3), padding='same')(encoder_pool4)
    center = layers.BatchNormalization()(center)    
    center = layers.Activation('relu')(center)
    center = layers.Conv2D(1024, (3, 3), padding='same')(center)
    center = layers.BatchNormalization()(center)
    center = layers.Activation('relu')(center)
    
    # Decoder4 decoder4 = decoder_block(center, encoder4, 512) # 16 
    decoder4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    decoder4 = layers.concatenate([encoder4, decoder4], axis=-1)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation('relu')(decoder4)
    decoder4 = layers.Conv2D(512, (3, 3), padding='same')(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation('relu')(decoder4)
    decoder4 = layers.Conv2D(512, (3, 3), padding='same')(decoder4)
    decoder4 = layers.BatchNormalization()(decoder4)
    decoder4 = layers.Activation('relu')(decoder4)
    
    # Decoder3 decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    decoder3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(decoder4)
    decoder3 = layers.concatenate([encoder3, decoder3], axis=-1)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation('relu')(decoder3)
    decoder3 = layers.Conv2D(256, (3, 3), padding='same')(decoder3)
    decoder3 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation('relu')(decoder3)
    decoder3 = layers.Conv2D(256, (3, 3), padding='same')(decoder3)
    decoder3_activation17 = layers.BatchNormalization()(decoder3)
    decoder3 = layers.Activation('relu')(decoder3_activation17)
    
    # Decoder2 decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    decoder2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(decoder3)
    decoder2 = layers.concatenate([encoder2, decoder2], axis=-1)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation('relu')(decoder2)
    decoder2 = layers.Conv2D(128, (3, 3), padding='same')(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation('relu')(decoder2)
    decoder2 = layers.Conv2D(128, (3, 3), padding='same')(decoder2)
    decoder2 = layers.BatchNormalization()(decoder2)
    decoder2 = layers.Activation('relu')(decoder2)
    
    # Decoder1 decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    decoder1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(decoder2)
    decoder1 = layers.concatenate([encoder1, decoder1], axis=-1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation('relu')(decoder1)
    decoder1 = layers.Conv2D(64, (3, 3), padding='same')(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation('relu')(decoder1)
    decoder1 = layers.Conv2D(64, (3, 3), padding='same')(decoder1)
    decoder1 = layers.BatchNormalization()(decoder1)
    decoder1 = layers.Activation('relu')(decoder1)
    
    # Decoder0 decoder0 = decoder_block(decoder1, encoder0, 32) # 256
    decoder0_conv2d_transpose_4 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(decoder1)
    decoder0 = layers.concatenate([encoder0, decoder0_conv2d_transpose_4], axis=-1)
    decoder0 = layers.BatchNormalization()(decoder0)
    decoder0 = layers.Activation('relu')(decoder0)
    decoder0_conv2d_20 = layers.Conv2D(32, (3, 3), padding='same')(decoder0)
    decoder0 = layers.BatchNormalization()(decoder0_conv2d_20)
    decoder0_activation_25 = layers.Activation('relu')(decoder0)
    decoder0_conv2d_21 = layers.Conv2D(32, (3, 3), padding='same')(decoder0_activation_25)
    decoder0 = layers.BatchNormalization()(decoder0_conv2d_21)
    decoder0_activation26 = layers.Activation('relu')(decoder0)
    
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(decoder0_activation26)
    
    layer = decoder0_conv2d_20
    
    
    l2_loss = tf.norm(X0 - layer, 'euclidean')/tf.norm(X0,'euclidean')
#    ra_loss = tf.norm(x_image - tf.reduce_mean(tf.convert_to_tensor(x_image)), 'euclidean')
    total_variation_loss = tf.reduce_sum(tf.image.total_variation(tf.convert_to_tensor(x_image + X)))
#    sigma_tv = 5e-7
    sigma_tv = 100
    loss = l2_loss + sigma_tv*total_variation_loss
    train_step=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(X[0][0][0][0]))
        for i in range(5000):
            _, loss_value = sess.run([train_step,loss])
            print("Loss: ", loss_value, 'Epoch:', i)
        # Get the image after the iterations
        O=sess.run(X)
    
    new_name = x_test_filenames[x].split('/')
    new_name = new_name[8]
    
    ar = np.squeeze(O)
    ar_r = ar[:,:,0]
    ar_g = ar[:,:,1]
    ar_b = ar[:,:,2]
    
    ar_norm_r = ((ar_r - np.min(ar_r.ravel()))/(np.max(ar_r.ravel())-np.min(ar_r.ravel())))*255
    ar_norm_g = ((ar_g - np.min(ar_g.ravel()))/(np.max(ar_g.ravel())-np.min(ar_g.ravel())))*255
    ar_norm_b = ((ar_b - np.min(ar_b.ravel()))/(np.max(ar_b.ravel())-np.min(ar_b.ravel())))*255
    
    im = np.stack([ar_norm_r,ar_norm_g, ar_norm_b])
    im = im.astype('uint8')
    io.imsave(save_loc + new_name,np.stack([ar_norm_r,ar_norm_g, ar_norm_b], axis = 2))
    io.imsave(raw_save + new_name, ar)