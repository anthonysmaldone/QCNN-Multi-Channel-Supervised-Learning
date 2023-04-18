# import packages
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras import datasets, layers, models
from sklearn.utils import shuffle

import cirq
from cirq.contrib.svg import SVGCircuit

import sympy
import numpy as np
import collections

import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from datetime import date
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
import os
import cv2

def datasize(datatype):

    if datatype=="COLORS":
        train_size = 2560
        test_size = 640

    if datatype == "COLORS_SHAPE":
        train_size = 5120
        test_size = 1280

    if datatype == "CHANNELS":
        train_size = 1000
        test_size = 200

    if datatype == "CIFAR10":
        train_size = 5000
        test_size = 1000

    if datatype == "MNIST":
        train_size = 5000
        test_size = 1000
    
    return train_size, test_size

def build_model_datasets(datatype):


    resize_x = 10 
    resize_y = 10
    global_batch_size = 50


    if datatype == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0
        print("Number of original training examples:", len(x_train))
        print("Number of original test examples:", len(x_test))
        print("Shape of training image data:", x_train.shape)
        x_train, x_test = tf.transpose(x_train, perm=[1,2,0]), tf.transpose(x_test, perm=[1,2,0])
        print("Shape of training image data after permutation:", x_train.shape)
        x_train, x_test = tf.image.resize(x_train[:,:,:datasize(datatype)[0]], (resize_x,resize_y)).numpy(), tf.image.resize(x_test[:,:,:datasize(datatype)[1]], (resize_x,resize_y)).numpy()
        x_train, x_test = tf.cast(x_train, tf.float32), tf.cast(x_test, tf.float32)
        x_train, x_test = tf.transpose(x_train, perm=[2,0,1]), tf.transpose(x_test, perm=[2,0,1])
        print("Shape of training image data after re-permuting:", x_train.shape)
        x_train, x_test = x_train[:datasize(datatype)[0]], x_test[:datasize(datatype)[1]]
        y_train, y_test = y_train[:datasize(datatype)[0]], y_test[:datasize(datatype)[1]]
        
        return x_train, x_test, y_train, y_test

    if datatype == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0
        print("Number of original training examples:", len(x_train))
        print("Number of original test examples:", len(x_test))
        print("Shape of training image data:", x_train.shape)
        x_train, x_test = tf.image.resize(x_train[:,:,:datasize(datatype)[0]], (resize_x,resize_y)).numpy(), tf.image.resize(x_test[:,:,:datasize(datatype)[1]], (resize_x,resize_y)).numpy()
        x_train, x_test = x_train[:datasize(datatype)[0]], x_test[:datasize(datatype)[1]]
        y_train, y_test = y_train[:datasize(datatype)[0]], y_test[:datasize(datatype)[1]]
        
        return x_train, x_test, y_train, y_test

    if datatype == "COLORS" or datatype == "COLORS_SHAPE":
        if datatype == "COLORS":
            data_dir = './colors'
        if datatype == "COLORS_SHAPE":
            data_dir = './colors_shapes'

        train = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            seed=42,
            batch_size = global_batch_size,
            subset="training",
            shuffle=True,
            image_size=(resize_x, resize_y))

        test = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            seed=42,
            batch_size = global_batch_size,
            subset="validation",
            shuffle=True,
            image_size=(resize_x, resize_y))
        
        return train, test
    if datatype == "CHANNELS":
        #Specify desired number of channels and classes
        channels = 12 
        classes = 10
        classes_to_add_to = channels-classes+1
        
        train_class_size = int(datasize(datatype)[0]/classes)
        test_class_size = int(datasize(datatype)[1]/classes)
        
        x_train = np.array([], dtype=np.int64).reshape(0,resize_x,resize_y,channels)
        y_train = np.array([]*train_class_size)
        x_test = np.array([], dtype=np.int64).reshape(0,resize_x,resize_y,channels)
        y_test = np.array([]*test_class_size)

        #Create synthetic training and testing data and labels
        for i in range(classes):
            x_training_class = np.random.rand(train_class_size,resize_x,resize_y,channels)
            y_training_class = np.array([i]*train_class_size)
            x_test_class = np.random.rand(test_class_size,resize_x,resize_y,channels)
            y_test_class = np.array([i]*test_class_size)

            for j in range(classes_to_add_to):
                x_training_class[...,i+j] += 0.5 
                x_test_class[...,i+j] += 0.5 
            x_train = np.concatenate((x_train,x_training_class))
            y_train = np.concatenate((y_train,y_training_class))
            x_test = np.concatenate((x_test,x_test_class))
            y_test = np.concatenate((y_test,y_test_class))

        #shuffle the newly generated data
        x_train,y_train = shuffle(x_train,y_train)
        x_test,y_test = shuffle(x_test,y_test)
        
        return x_train, x_test, y_train, y_test