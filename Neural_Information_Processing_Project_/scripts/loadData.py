# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:14:13 2019

@author: Justi
"""


#---------------------Libraries--------------------#
import functools
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

#---------------------Tensorflow Libraries--------------------#
import tensorflow as tf
import tensorflow.contrib as tfcontrib
keep_prob = tf.placeholder("float")

# %%

class dataLoader():
    
    def __init__(self, img_shape,
                 x_train_filenames, 
                 y_train_filenames,
                 x_val_filenames, 
                 y_val_filenames, 
                 batch_size,
                 scale = 1/255.,
                 rgbgray = False,
                 h_flip = True,
                 hue_delta = False, 
                 w_shift = False, 
                 h_shift = False):
        
        self.img_shape = img_shape
        self.x_train_filenames = x_train_filenames
        self.y_train_filenames = y_train_filenames
        self.x_val_filenames = x_val_filenames
        self.y_val_filenames = y_val_filenames
        self.batch_size = batch_size
        self.scale = scale
        self.hue_delta = hue_delta
        self.rgbgray = rgbgray
        self.h_flip = h_flip
        self.w_shift = w_shift
        self.h_shift = h_shift
        
    
    def shift_img(self, output_img, label_img,img_shape, width_shift_range, height_shift_range):
        """This fn will perform the horizontal or vertical shift"""
        if width_shift_range or height_shift_range:
            if width_shift_range:
                width_shift_range = tf.random_uniform([], 
                                                  -width_shift_range * img_shape[1],
                                                  width_shift_range * img_shape[1])
            if height_shift_range:
                height_shift_range = tf.random_uniform([],
                                                   -height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
        # Translate both 
        output_img = tfcontrib.image.translate(output_img,
                                                 [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(label_img,
                                                 [width_shift_range, height_shift_range])
        
        return output_img, label_img


    def flip_img(self, horizontal_flip, tr_img, label_img):
      if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
      return tr_img, label_img
    
    def _augment(self,
                 img,
                 label_img,
                 img_shape,
                 resize=None,  # Resize the image to some size e.g. [256, 256]
                 scale=1,
                 gray = False,  # Scale image e.g. 1 / 255.
                 hue_delta=0,  # Adjust the hue of an RGB image by random factor
                 horizontal_flip=False,  # Random left right flip,
                 width_shift_range=0,  # Randomly translate the image horizontally
                 height_shift_range=0): # Randomly translate the image vertically ):
        
        if resize is not None:
            # Resize both images
            label_img = tf.image.resize_images(label_img, resize)
            img = tf.image.resize_images(img, resize)
            
        if hue_delta:
            img = tf.image.random_hue(img, hue_delta)
            
        if gray:
            img = tf.image.rgb_to_grayscale(img)
            
        img, label_img = self.flip_img(horizontal_flip, img, label_img)
        img, label_img = self.shift_img(img, label_img,img_shape, width_shift_range, height_shift_range)
        label_img = tf.to_float(label_img) * scale
        img = tf.to_float(img) * scale 
        
        return img, label_img
    
    def _process_pathnames(self, fname, label_path):
      img_str = tf.read_file(fname)
      img = tf.image.decode_png(img_str, channels = 0)
      label_img_str = tf.read_file(label_path)
      label_img = tf.image.decode_png(label_img_str, channels = 0)
      
      return img, label_img
    
    def get_baseline_dataset(self,filenames, labels, img_shape, batch_size, shuffle=True, preproc_fn= functools.partial(_augment), threads=5):
        num_x = len(filenames)
        print('Length of Data:',num_x)
        # Create a dataset from the filenames and labels
        # Map our preprocessing function to every element in our dataset, taking
        # advantage of multithreading
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._process_pathnames, num_parallel_calls = threads)
        if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
            assert batch_size == 1, "Batching images must be of the same size"
    
        dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
        print('Reached Mapping')
        
        if shuffle:
            dataset = dataset.shuffle(num_x)
            
        # It's necessary to repeat our data for all epochs 
        dataset = dataset.repeat().batch(batch_size)
        
        return dataset

        
    def setParams(self):
        
        # Set up train and validation datasets
        if self.hue_delta == True:
            hue_scale = 0.1
        else:
            hue_scale = 0
        
        if self.w_shift == True:
            wrange = 0.1
        else:
            wrange = 0
        
        if self.h_shift == True:
            hrange = 0.1
        else:
            hrange = 0
            
        if self.rgbgray == True:
            convert_gray = True
        else:
            convert_gray = False
            
        tr_cfg = {
                'resize': [self.img_shape[0], self.img_shape[1]],
                'img_shape':self.img_shape,
                'scale': self.scale,
                'gray': convert_gray,
                'hue_delta': hue_scale,
                'horizontal_flip': self.h_flip,
                'width_shift_range': wrange,
                'height_shift_range': hrange,
                }
        tr_preprocessing_fn = functools.partial(self._augment, **tr_cfg)
    
        val_cfg = {
                'resize': [self.img_shape[0], self.img_shape[1]],
                'img_shape':self.img_shape,
                'scale': self.scale,
    			'gray': convert_gray,
                }
        val_preprocessing_fn = functools.partial(self._augment, **val_cfg)
        
        train_ds = self.get_baseline_dataset(self.x_train_filenames,
                                    self.y_train_filenames,
                                    self.img_shape,
                                    self.batch_size,
                                    preproc_fn=tr_preprocessing_fn)
        val_ds = self.get_baseline_dataset(self.x_val_filenames,
                                  self.y_val_filenames,
                                  self.img_shape,
                                  self.batch_size, 
                                  preproc_fn=val_preprocessing_fn)
        
        return train_ds, val_ds
        


    

