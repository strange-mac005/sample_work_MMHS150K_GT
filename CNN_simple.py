# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:49:01 2021

@author: k1d14
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#import keras 

    
data=np.load('data.npz',allow_pickle=True)

X=data['arr_0']
y=data['arr_1']
df=pd.read_csv("outputt.csv").to_numpy()
convert = OneHotEncoder(sparse=False).fit(np.reshape(df[:,5],(-1,1)))
y=convert.transform(np.reshape(df[:,5],(149823,-1)))
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)

  #X_trainnp.resize=(104876,80,80,1)
X_train=np.resize(X_train,(104876,80,80,1))
X_test=np.resize(X_test,(44947, 80, 80,1))

print(X_test.shape)


## Part 2 - Building the CNN

### Initialising the CNN
cnn = tf.keras.models.Sequential()

### Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', input_shape=[80, 80, 1]))

### Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

### Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

### Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units=6, activation='softmax'))

## Part 3 - Training the CNN

### Compiling the CNN

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

tf.keras.utils.plot_model(cnn,show_shapes=True,to_file="cnn__model.png")

### Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = X_train,y=y_train, epochs = 1)

## Part 4 - Making a multiple prediction

result = cnn.predict(X_test)

