# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:52:26 2019

@author: Justi
"""

# %%
#---------------------Libraries--------------------#
import numpy as np
import os
import datetime
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

#---------------------Tensorflow Libraries--------------------#
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
# %%

class Unet():
    
    def __init__(self, batch_size, epochs, optimizer, num_train_examples, num_val_examples, log_path, base_path):
        
        self.img_rows = 256
        self.img_cols = 256
        self.channels= 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.log_path = log_path
        self.base_path = base_path
        self.model = self.build_unet()
        
    def build_unet(self):
        
        def conv_block(input_tensor, num_filters):
            """
            Function that defines the convolution layer
        
            Parameters
            ----------
            input_tensor = input data
            num_filters = number of filters
            
        
            Returns
            -------
            encoder = applies encoding operation
            
        
            """
            encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
            encoder = layers.BatchNormalization()(encoder)    
            encoder = layers.Activation('relu')(encoder)
            encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation('relu')(encoder)
            
            return encoder
    
        def encoder_block(input_tensor, num_filters):
            """
            Function that defines the encoder operation
        
            Parameters
            input_tensor = input data 
            num_filters = number of filters
            ----------
            
        
            Returns
            -------
            encoder_pool = max pooling operation 
            encoder = applies encoding operation
        
            """
            
            encoder = conv_block(input_tensor, num_filters)
            encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
            
            return encoder_pool, encoder
    
        def decoder_block(input_tensor, concat_tensor, num_filters):
            
            """
            Function that defines the decoder operation
        
            Parameters
            ----------
            input_tensors = input data
            concat_tensor = tensor from previous layer
            num_filters = number of filters
            
        
            Returns
            -------
            decoder = applies decoding operation
            
        
            """
            
            decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
            decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation('relu')(decoder)
            decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation('relu')(decoder)
            decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation('relu')(decoder)
            
            return decoder
        
        inputs = layers.Input(shape=self.img_shape)
        encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
        encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
        encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
        encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
        encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
        center = conv_block(encoder4_pool, 1024) # center
        decoder4 = decoder_block(center, encoder4, 512) # 16
        decoder3 = decoder_block(decoder4, encoder3, 256) # 32
        decoder2 = decoder_block(decoder3, encoder2, 128) # 64
        decoder1 = decoder_block(decoder2, encoder1, 64) # 128
        decoder0 = decoder_block(decoder1, encoder0, 32) # 256
        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(decoder0)
        
        model = models.Model(inputs=[inputs], outputs=[outputs])
        
        return model
    
    
    def train(self, train_ds, val_ds):
        
        def dice_coeff(y_true, y_pred):
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
        
        def getSavePath(base_path):
            
            currentDT = datetime.datetime.now()
            weights_name = 'weights_' + str(currentDT.year) + '_' + str(currentDT.month) + '_' + str(currentDT.day) + '_' + str(currentDT.hour) + '_' + str(currentDT.minute) + '_' + str(currentDT.second)
            save_model_path =  base_path +  weights_name + '_' + 'weights.{epoch:02d}--{val_loss:.2f}.hdf5'
            
            return save_model_path
        
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss= bce_dice_loss, metrics=[dice_loss, dice_coeff])
        
        try:
            os.mkdir(self.log_path)
        except FileExistsError:
            
            print('Folder already exists')
            
        try:
            os.mkdir(self.base_path)
        except FileExistsError:
            print('Folder already exists')
        
        log_dir = self.log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_model_path = getSavePath(self.base_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
        
        history = self.model.fit(train_ds,
                                   steps_per_epoch=int(np.ceil(self.num_train_examples / float(self.batch_size))),
                                   epochs=self.epochs, 
                                   validation_data=val_ds,
                                   validation_steps=int(np.ceil(self.num_val_examples / float(self.batch_size))),
                                   callbacks=[cp,tensorboard_callback])