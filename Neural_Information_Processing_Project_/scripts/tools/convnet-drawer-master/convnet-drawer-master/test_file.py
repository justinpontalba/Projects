# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:39:43 2019

@author: Justi
"""

from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx
from matplotlib_util import save_model_to_file

model = Model(input_shape=(51, 51, 3))
model.add(Conv2D(25, (48, 48), (4, 4)))
model.add(MaxPooling2D((2, 2), strides=(7, 7)))
model.add(Conv2D(50, (20, 20), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(7, 7)))
model.add(Conv2D(80, (5, 5), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(7, 7)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(1024))

save_model_to_pptx(model, "CNN3.pptx")