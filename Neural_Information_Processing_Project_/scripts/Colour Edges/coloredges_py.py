# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:39:25 2019

@author: Justi
"""

import numpy as np
from skimage.filters import sobel_h, sobel_v, sobel
from skimage import io
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim


def colouredges_py(img):
    
    img = img.astype('float32')/255

    rx = sobel_h(img[:,:,0])
    gx = sobel_h(img[:,:,1])
    bx = sobel_h(img[:,:,2])
    
    ry = sobel_v(img[:,:,0])
    gy = sobel_v(img[:,:,1])
    by = sobel_v(img[:,:,2])
    
    Jx = rx**2 + gx**2 + bx**2
    Jy = ry**2 + gy**2 + by**2
    Jxy = rx*ry + gx*gy + bx*by
    
    D = np.sqrt(np.abs(Jx**2 - 2*Jx*Jy + Jy**2 + 4*Jxy**2))
    
    e1 = (Jx + Jy + D)/2
    
    edge_magnitude = np.sqrt(e1)
    
    edge_orientation = np.arctan2(-1*Jxy, e1 - Jy)
    
    edge_magnitude = (edge_magnitude/np.max(edge_magnitude.ravel()))*255
    edge_orientation = (edge_orientation/np.max(edge_orientation.ravel()))*255
    
    return edge_magnitude, edge_orientation

def colourEdges_gw(img, edge_mag = True):
	
	r = img[:,:,0]
	g = img[:,:,1]
	b = img[:,:,2]
	
	plt.figure()
	plt.imshow(img)
	plt.title('Input Image')
	
	plt.figure()
	plt.subplot(131)
	plt.imshow(r, cmap = 'gray')
	plt.title('r')
	
	plt.subplot(132)
	plt.imshow(g, cmap = 'gray')
	plt.title('g')
	
	plt.subplot(133)
	plt.imshow(b, cmap = 'gray')
	plt.title('b')
	
	gxx_r = (sobel_h(r))
	gxx_g = sobel_h(g)
	gxx_b = sobel_h(b)
	
	gyy_r = sobel_v(r)    
	gyy_g = sobel_v(g)
	gyy_b = sobel_v(b)
	
	gxy_r = np.abs(gxx_r) + np.abs(gyy_r)
	gxy_g = np.abs(gxx_g) + np.abs(gyy_g)
	gxy_b = np.abs(gxx_b) + np.abs(gyy_b)
	
	gxx = np.square(np.abs(gxx_r)) + np.square(np.abs(gxx_g)) + np.square(np.abs(gxx_b))
	gyy = np.square(np.abs(gyy_r)) + np.square(np.abs(gyy_g)) + np.square(np.abs(gyy_b))
	
	gxy = gxx_r*gyy_r + gxx_g*gyy_g + gxx_b*gyy_b 
	
	theta = (1/2)*np.arctan2((2*gxy),(gxx-gyy))
	mag = (0.5*((gxx + gyy) + (gxx - gyy)*np.cos(2*theta) + 2*gxy*np.sin(2*theta)))**0.5
	
	if edge_mag == True:
		mag = np.sqrt(np.abs(gxx**2 - 2*gxx * gyy + gyy**2 + 4*gxy**2))
		ei = (gxx + gyy + mag)/2
		mag = np.sqrt(ei)
	
	plt.figure()
	plt.subplot(131)
	plt.imshow(gxy_r,cmap = 'gray')
	plt.title('R-Edges')

	plt.subplot(132)
	plt.imshow(gxy_g,cmap = 'gray')
	plt.title('G-Edges')
	
	plt.subplot(133)
	plt.imshow(gxy_b,cmap = 'gray')
	plt.title('B-Edges')
	
	plt.figure()
	plt.imshow(mag,cmap = 'gray')
	plt.title('Colour-Edges')
	
	
	return gxy_r,gxy_g,gxy_b,theta, mag
	
	
	
	
	