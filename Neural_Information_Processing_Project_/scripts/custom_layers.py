# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:54:15 2019

@author: jpontalb
"""
import tensorflow as tf
from tensorflow.keras import layers
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from functions import tensorwrapper


def colourEdges(inputTensor, edge_mag = True):
	
	gradients = tf.image.sobel_edges(inputTensor)
	
	r_dy = gradients[0,:,:,0,0]
	r_dx = gradients[0,:,:,0,1]
	g_dy = gradients[0,:,:,1,0]
	g_dx = gradients[0,:,:,1,1]
	b_dy = gradients[0,:,:,2,0]
	b_dx = gradients[0,:,:,2,1]
	

#	gxy_r = tf.abs(r_dx) + tf.abs(r_dy)
#	gxy_g = tf.abs(g_dx) + tf.abs(g_dy)
#	gxy_b = tf.abs(b_dx) + tf.abs(b_dy)
	
	gxx = tf.square(tf.abs(r_dx)) + tf.square(tf.abs(g_dx)) + tf.square(tf.abs(b_dx))
	gyy = tf.square(tf.abs(r_dy)) + tf.square(tf.abs(g_dy)) + tf.square(tf.abs(b_dy))
	
	gxy = r_dx*r_dy + g_dx*g_dy + b_dx*b_dy 
	
	theta = (1/2)*tf.atan2((2*gxy),(gxx-gyy))
	mag = (0.5*((gxx + gyy) + (gxx - gyy)*tf.cos(2*theta) + 2*gxy*tf.sin(2*theta)))**0.5
	
	if edge_mag == True:
		mag = tf.sqrt(tf.abs(gxx**2 - 2*gxx * gyy + gyy**2 + 4*gxy**2))
		ei = (gxx + gyy + mag)/2
		mag = tf.sqrt(ei)
	
	
	mag = tf.reshape(mag,(1,256,256,1))
	
	return mag
