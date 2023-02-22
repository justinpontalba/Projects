# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 23:43:56 2019

@author: giora
"""

import math
from keras import layers
from keras.models import Model
from custom_layers import colourEdges
import tensorflow as tf

class UNet:               
    '''
    Description:
    Class that creates a UNet model architecture. The model can be
    accessed by the object via the model member variable. Any other 
    number of parameters can be changed: image shape, initializer, 
    kernel/filter size, activation function and a batch normalization flag.
           
    Example(s): 
    >> unet = UNet()
    >> model = unet.get_model()
    
    >> unet = UNet(batch_norm=True, kernel_size=(5, 5))
    >> model = unet.get_model()
    
    Member variables:
    - depth(int): determines the number of encoding/decoding levels -> default: 4
    - img_shape(tuple): dimensions of input image -> default: (256, 256, 1)
    - initializer(str): weight initialization method -> default: 'he_normal'
    - kernel_size(tuple): size of convolution filter kernels -> default: (3, 3)
    - num_filters(int): initial number of filters for the 2-D convolutions -> default: 64
    - activation(str): activation functions used in convolution blocks -> default: 'relu'
    - activation_output(str): activation function used in output layer -> default: 'sigmoid'
    - batch_norm(bool): flag to determine whether or not batch normalization is used -> default: False
    - pretrained_weights(str): directory of model containing pretrained weights -> default: None

    Member functions:
    - get_model: returns the keras model
    - get_parameters: displays the parameters of the UNet        
    - __conv_block: returns result of convolution activation, convolution, activation -> optional batch normalization layers before activations
    - __encoder_block: returns the result of a convolution block followed by max-pooling operation
    - __decoder_block: returns results of transposed convolution, concatenation, activation, followed by a convolution block -> optional batch normalization layers before activations
    - __get_model: returns the final keras model used for training
    
    '''        
    def __init__(self, depth=4, img_shape = (256,256,1), initializer='he_normal', kernel_size=(3, 3), num_filters=64, activation='relu', activation_output='sigmoid', batch_norm=False, pretrained_weights=None):
        
            
        self.img_shape = img_shape
        self.depth = depth 
        n = int(math.log(img_shape[0],2))
        assert self.depth >= 1 and self.depth <= n, "The depth must be between 1 and {0}.".format(n)     
        del n                      
        self.initializer = initializer
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_output = activation_output
        self.batch_norm = batch_norm 
        self.pretrained_weights = pretrained_weights
        self.num_filters = num_filters
        
    def get_model(self):
        return self.__forward(self.depth)
 
    def get_parameters(self, disp=True):
        if(disp):
            print('Depth:', self.depth)
            print('Model:', 'U-Net')
            print('Input shape:', self.img_shape)
            print('Initializer:', self.initializer) 
            print('Kernel size:', self.kernel_size)
            print('Number of filters:', self.num_filters)
            print('Activation:', self.activation) 
            print('Activation (output):', self.activation_output)
            print('Batch normalization:', self.batch_norm) 
            print('Pretrained weights:', self.pretrained_weights, '\n') 
        
        params = {'Input shape': self.img_shape, 'Initializer': self.initializer,
                  'Kernel size': self.kernel_size, 'Activation': self.activation,
                  'Activation(output)': self.activation_output, 
                  'Number of filters': self.num_filters,
                  'Batch normalization': self.batch_norm, 
                  'Pretrained weights': self.pretrained_weights,
                  'Model': 'U-Net', 'Depth': self.depth}    
        return params

    def __conv_block(self, input_tensor, num_filters):
        if(self.batch_norm):
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(self.activation)(encoder)
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(encoder)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(self.activation)(encoder)            
        else:
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            encoder = layers.Activation(self.activation)(encoder)
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(encoder)
            encoder = layers.Activation(self.activation)(encoder)             
        return encoder   

    def __encoder_block(self, input_tensor, num_filters):
        encoder = self.__conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)         
        return encoder_pool, encoder
    
    def __edgeBlock(self, input_tensor, num_filters):
        
        edges = colourEdges(input_tensor)
        conv1 = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(edges)
        batch1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(batch1)
        batch2 = layers.BatchNormalization()(conv2)
        
        return batch2

    def __decoder_block(self, input_tensor, concat_tensor, num_filters):   
        if(self.batch_norm):
            decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
            decoder = layers.concatenate([concat_tensor, decoder], axis=-1)        
            decoder = layers.BatchNormalization()(decoder)            
            decoder = layers.Activation(self.activation)(decoder)
            decoder = self.__conv_block(decoder, num_filters)                       
        else:
            decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
            decoder = layers.concatenate([concat_tensor, decoder], axis=-1)                    
            decoder = layers.Activation(self.activation)(decoder)
            decoder = self.__conv_block(decoder, num_filters)                 
        return decoder

    def __forward(self, depth):
        encoders, decoders, encoders_pool = [], [], []
        input_layer = layers.Input(shape=self.img_shape)
        edges = self.__edgeBlock(input_layer, self.num_filters)
        encoder_pool, encoder = self.__encoder_block(input_layer, self.num_filters)       
        encoders.append(encoder) 
        encoders_pool.append(encoder_pool)    

        # Encoding path
        for i in range(depth-1):
            encoder_pool, encoder = self.__encoder_block(encoders_pool[i], (2**(i+1))*self.num_filters)
            encoders.append(encoder)   
            encoders_pool.append(encoder_pool)
			
        center = self.__conv_block(encoders_pool[-1], (2**(depth))*self.num_filters)       
        decoder = self.__decoder_block(center, encoders[-1], (2**(depth-1))*self.num_filters)
        decoders.append(decoder)

        # Decoding path
        for j in range(depth-1):            
            decoder = self.__decoder_block(decoders[j], encoders[-(j+2)], (2**(depth-(j+2)))*self.num_filters)
            decoders.append(decoder)

        output_layer = layers.Conv2D(3, (1, 1), activation=self.activation_output)(decoders[-1])
        model = Model(inputs=[input_layer], outputs=[output_layer])             
        # Pretrained weights
        if(self.pretrained_weights):
            model.load_weights(self.pretrained_weights)            
        return model

#---------------------------------------------------------------------------

class UResNet:               
    '''
    Description:
    Class that creates a UResNet model architecture. The convolution blocks 
    from the original U-net have been replaced with residual blocks. The 
    model can be accessed by the object via the model member variable. Any 
    other number of parameters can be changed: image shape, initializer, 
    kernel/filter size, activation function and a batch normalization flag.
           
    Example(s): 
    >> uresnet = UResNet()
    >> model = uresnet.get_model()
    
    >> uresnet = UResNet(batch_norm=True, kernel_size=(5, 5))
    >> model = uresnet.get_model()
    
    Member variables:
    - img_shape(tuple): dimensions of input image -> default is (256, 256, 1)
    - initializer(str): weight initialization method -> default is 'he_normal'
    - kernel_size(tuple): size of convolution filter kernels -> default is (3, 3)
    - num_filters(int): initial number of filters for the 2-D convolutions
    - activation(str): activation functions used in convolution blocks -> default is 'relu'
    - activation_output(str): activation function used in output layer -> default is 'sigmoid'
    - batch_norm(bool): flag to determine whether or not batch normalization is used -> default is False
    - pretrained_weights(str): directory of model containing pretrained weights -> None

    Member functions:
    - get_model: returns the keras model
    - get_parameters: displays the parameters of the UNet        
    - __res_block: returns result of convolution activation, convolution, activation concatenated with the input -> optional batch normalization layers before activations
    - __encoder_block: returns the result of a convolution block followed by max-pooling operation
    - __decoder_block: returns results of transposed convolution, concatenation, activation, followed by a convolution block -> optional batch normalization layers before activations
    - __get_model: returns the final keras model used for training
    
    '''        
    def __init__(self, img_shape=(256, 256, 1), initializer='he_normal', kernel_size=(3, 3), num_filters=32, activation='relu', activation_output='sigmoid', batch_norm=False, pretrained_weights=None):
                        
        self.img_shape = img_shape                             
        self.initializer = initializer
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_output = activation_output
        self.batch_norm = batch_norm 
        self.pretrained_weights = pretrained_weights
        self.num_filters = num_filters
        
    def get_model(self):
        return self.__forward()
 
    def get_parameters(self, disp=True):
        if(disp):
            print('Model:','UResNet')
            print('Input shape:',self.img_shape)
            print('Initializer:',self.initializer) 
            print('Kernel size:',self.kernel_size)
            print('Number of filters:',self.num_filters)
            print('Activation:',self.activation) 
            print('Activation (output):',self.activation_output)
            print('Batch normalization:',self.batch_norm) 
            print('Pretrained weights:',self.pretrained_weights, '\n') 
        
        params = {'Input shape': self.img_shape, 'Initializer': self.initializer,
                  'Kernel size': self.kernel_size, 'Activation': self.activation,
                  'Activation(output)': self.activation_output, 
                  'Number of filters': self.num_filters,
                  'Batch normalization': self.batch_norm, 
                  'Pretrained weights': self.pretrained_weights,
                  'Model':'UResNet'}       
        return params

    def __res_block(self, input_tensor, num_filters):
        if(self.batch_norm):
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(self.activation)(encoder)
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(encoder)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(self.activation)(encoder)  
            encoder = layers.concatenate([input_tensor, encoder], axis=-1)
        else:
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            encoder = layers.Activation(self.activation)(encoder)
            encoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(encoder)
            encoder = layers.Activation(self.activation)(encoder)   
            encoder = layers.concatenate([input_tensor, encoder], axis=-1)
        return encoder   

    def __encoder_block(self, input_tensor, num_filters):
        encoder = self.__res_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)        
        return encoder_pool, encoder

    def __decoder_block(self, input_tensor, concat_tensor, num_filters):   
        if(self.batch_norm):
            decoder_input = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
            decoder = layers.concatenate([concat_tensor, decoder_input], axis=-1)        
            decoder = layers.BatchNormalization()(decoder)            
            decoder = layers.Activation(self.activation)(decoder)
            decoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(decoder)               
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation(self.activation)(decoder)
            decoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(decoder)        
            decoder = layers.BatchNormalization()(decoder)            
            decoder = layers.Activation(self.activation)(decoder)    
            decoder = layers.concatenate([decoder_input, decoder], axis=-1)
        else:
            decoder_input = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
            decoder = layers.concatenate([concat_tensor, decoder_input], axis=-1)                    
            decoder = layers.Activation(self.activation)(decoder)
            decoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(decoder)               
            decoder = layers.Activation(self.activation)(decoder)
            decoder = layers.Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(decoder)                   
            decoder = layers.Activation(self.activation)(decoder)  
            decoder = layers.concatenate([decoder_input, decoder], axis=-1)                  
        return decoder

    def __forward(self):
        input_layer = layers.Input(shape=self.img_shape)
        encoder0_pool, encoder0 = self.__encoder_block(input_layer, self.num_filters)
        encoder1_pool, encoder1 = self.__encoder_block(encoder0_pool, 2*self.num_filters)
        encoder2_pool, encoder2 = self.__encoder_block(encoder1_pool, 4*self.num_filters)
        encoder3_pool, encoder3 = self.__encoder_block(encoder2_pool, 8*self.num_filters)
        encoder4_pool, encoder4 = self.__encoder_block(encoder3_pool, 16*self.num_filters)
        center = self.__res_block(encoder4_pool, 32*self.num_filters)
        decoder4 = self.__decoder_block(center, encoder4, 16*self.num_filters)
        decoder3 = self.__decoder_block(decoder4, encoder3, 8*self.num_filters)
        decoder2 = self.__decoder_block(decoder3, encoder2, 4*self.num_filters)
        decoder1 = self.__decoder_block(decoder2, encoder1, 2*self.num_filters)
        decoder0 = self.__decoder_block(decoder1, encoder0, self.num_filters)
        output_layer = layers.Conv2D(3, (1, 1), activation=self.activation_output)(decoder0)
        model = Model(inputs=[input_layer], outputs=[output_layer])      
        # Pretrained weights
        if(self.pretrained_weights):
            model.load_weights(self.pretrained_weights)
        return model

#---------------------------------------------------------------------------

        
        
        
        
        
        
        
            