# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:41:32 2021

@author: k1d14
"""
import re
import cv2
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder


df=pd.read_csv("outputt.csv").to_numpy()
X = []
y = []
convert = OneHotEncoder(sparse=False).fit(np.reshape(df[:,5],(-1,1)))
for p in df:
        category = convert.transform(np.reshape(p[5],(1,-1)))
        t_id=re.sub('[^\d]','',p[-1])
        img_array = cv2.imread("G:\\MMHS150K\\img_resized\\" +t_id+".jpg",cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)
        print("Loaded G:\\MMHS150K\\img_resized\\" +t_id+".jpg")
np.savez("data.npz", X, y)