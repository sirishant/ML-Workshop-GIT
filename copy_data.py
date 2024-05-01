#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:35:49 2024

@author: sirishant
"""

path = 'Data'
dest = 'New_data/'
import os
import shutil
P = sorted(os.listdir(path))
for i in P:
    if os.path.isdir(path+'/'+i):
        new_path = path+'/'+i
        if not os.path.isdir(dest+i):
            os.makedirs(dest+i)
        Q = sorted(os.listdir(new_path))
        for j in range(100):
            shutil.copy(new_path+'/'+Q[j],dest+i)
            
        
        