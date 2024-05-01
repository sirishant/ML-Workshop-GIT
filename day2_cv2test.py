#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:05:48 2024

@author: sirishant
"""

import numpy as np
import cv2

# I = np.zeros((255,255))
I = np.ones((256,256,3), np.uint8)*255

# for i in range(1,255):
#     for j in range(1,255):
#         if i%2 == 0 or j%2 == 0:
#             I[i,j] = 255
#         else:
#             I[i,j] = 127

# for i in range(1,127):
#     for j in range(1,128):
#         I[i,j] = 255

# for i in range(1,127):
#     for j in range(128,255):
#         I[i,j] = 0
        
# for i in range(128,255):
#     for j in range(1,128):
#         I[i,j] = 127
        
# for i in range(128,255):
#     for j in range(128,255):
#         I[i,j] = 255

for i in range(1,128):
    for j in range(1,128):
        if j < i:
            I[i,j,0] = 0

for i in range (128,255):
    for j in range(128,255-i):
        I[i,j] = 0
        # else:
        #     I[i,j] = 0

for i in range (128,255):
    for j in range(1,128):
        if j > i-128:
            I[i,j,1] = 0
        else:
            I[i,j] = 0

for i in range (1,128):
    for j in range(128,255):
        if j-127 < i:
            I[i,j,2] = 0

np.uint8(I)

cv2.imwrite("test.png", I)