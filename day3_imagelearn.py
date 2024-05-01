#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:22:13 2024

@author: sirishant
"""

import cv2
import numpy as np

I = cv2.imread('Lena.png', 0)

# kernel = np.ones((10,10))/100
# kernel = [[1,0,-1],[2,0,-2],[1,0,-1]]
kernel = [[0,-1,0],[-1,5,-1],[0,-1,0]]
kernel = np.array(kernel)
# print(kernel)

new_I = cv2.filter2D(I, -1, kernel)
cv2.imwrite('Lena_convoluted3.png', new_I)