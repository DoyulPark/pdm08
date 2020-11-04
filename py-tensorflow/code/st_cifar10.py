import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

from keras.datasets import cifar10

# Load CIFAR10 dataset using tensorflow.keras
# Dividing data into training and test set
(trainX, trainy), (testX, testy) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Sidebar
st.sidebar.header('CIFAR10')
st.sidebar.subheader('Dataset of several images')
# Show a random images
if st.sidebar.checkbox('Show a random image from train CIFAR10'):
    num = np.random.randint(0, trainX.shape[0])
    image = trainX[idx]
    st.sidebar.image(image, caption=class_names[trainy[idx]], width=192)

if st.sidebar.checkbox('Show a random image from test CIFAR10'):
    num = np.random.randint(0, testX.shape[0])
    image = testX[idx]
    st.sidebar.image(image, caption=class_names[testy[idx]], width=192)

# Main 
st.title('DL using CNN2D')
st.header('Dataset: CIFAR10')
#spending a few lines to describe our dataset
st.text("""Dataset of 50,000 32x32x3 gray training images, 
        labeled over 0 to 9, 
        and 10,000 test images.""")

# Information of mnist dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
if st.checkbox('Show images sizes'):
    st.write(f'##### X Train Shape: {trainX.shape}') 
    st.write(f'##### X Test Shape: {testX.shape}')
    st.write(f'##### Y Train Shape: {trainy.shape}')
    st.write(f'##### Y Test Shape: {testy.shape}')

# display one random image from our training set:
st.subheader('Inspecting dataset')
if st.checkbox('Show random image from the train set'):
    num = np.random.randint(0, trainX.shape[0])
    image = trainX[idx]
    st.image(image, caption=class_names[trainy[num]], width=96) 
st.write('***')

if st.checkbox('Show random image from the test set'):
    num = np.random.randint(0, testX.shape[0])
    image = testX[idx]
    st.image(image, caption=class_names[testy[num]], width=96) 
st.write('***')

if st.checkbox('Show 10 different image from the train set'):
    num_10 = np.unique(trainy, return_index=True)[1]
#     st.write(num_10)
    images = trainX[num_10]
    fig =plt.figure(figsize=(10,6))
    for i in range(len(images)):
        # define subplot
        plt.subplot(2,5,1 + i) #, sharey=False)
        # plot raw pixel data
        plt.imshow(images[i])
        plt.title(class_names[i])
        plt.xticks([])
        plt.yticks([])

if st.checkbox('Show 10 different image from the test set'):
    num_10 = np.unique(trainy, return_index=True)[1]
#     st.write(num_10)
    images = testX[num_10]
    fig =plt.figure(figsize=(10,6))
    for i in range(len(images)):
        # define subplot
        plt.subplot(2,5,1 + i) #, sharey=False)
        # plot raw pixel data
        plt.imshow(images[i])
        plt.title(class_names[i])
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("10 different images in CIFAR-10", fontsize=18)
    st.pyplot()  # Warning




