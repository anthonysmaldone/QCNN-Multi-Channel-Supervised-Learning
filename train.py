# import packages
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras import datasets, layers, models
from sklearn.utils import shuffle

import cirq

import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from datetime import date
from tensorflow.keras.utils import plot_model

import os
import cv2



datatype = "COLORS"


num_of_epochs = 10 
global_learning_rate = 0.01
global_batch_size = 50
resize_x = 10 
resize_y = 10 


from prepare_data import datasize, build_model_datasets
import generate_output
import models

details = [datasize(datatype)[0],resize_x,resize_y,global_learning_rate,global_batch_size,datasize(datatype)[1],datatype,num_of_epochs]

#save_path = '/mnt/c/Users/Anthony M. Smaldone/Desktop/phase_based_QCNN/'

############################
DR_U1_QCNN = False 
DRR_U1_QCNN = False
MW_U1_QCNN = False 
control_U1_QCNN = False

DR_U2_QCNN = False 
DRR_U2_QCNN = False 
MW_U2_QCNN = False 
control_U2_QCNN = False

classical_model = True 
#############################


models_to_train = []


####################
if DR_U1_QCNN:
    models_to_train.append(models.DR_U1_QCNN_model(datatype))

if DRR_U1_QCNN:
    models_to_train.append(models.DRR_U1_QCNN_model(datatype))

if MW_U1_QCNN:
    models_to_train.append(models.QCNN_U1_weighted_control_model(datatype))

if control_U1_QCNN:
    models_to_train.append(models.QCNN_U1_control_model(datatype))

if DR_U2_QCNN:
    models_to_train.append(models.DR_U2_QCNN_model(datatype))

if DRR_U2_QCNN:
    models_to_train.append(models.DRR_U2_QCNN_model(datatype))

if MW_U2_QCNN:
    models_to_train.append(models.QCNN_U2_weighted_control_model(datatype))

if control_U2_QCNN:
    models_to_train.append(models.QCNN_U2_control_model(datatype))

if classical_model:
    models_to_train.append(models.CNN_classical_model(datatype))




def train_model(model_to_train):
    model = model_to_train
##########################
    model.summary()
############################

# compile quantum convolutional neural network and train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=global_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if datatype == "COLORS" or datatype == "COLORS_SHAPE":
        model_history = model.fit(build_model_datasets(datatype)[0], validation_data=build_model_datasets(datatype)[1], batch_size=global_batch_size, epochs=num_of_epochs)
    if datatype == "MNIST" or datatype == "CIFAR10" or datatype == "CHANNELS":
        model_history = model.fit(build_model_datasets(datatype)[0], build_model_datasets(datatype)[2], validation_data=(build_model_datasets(datatype)[1],build_model_datasets(datatype)[3]) , epochs=num_of_epochs, batch_size=global_batch_size)

    generate_output.save_output_imgs(model,model_history,details)




for x in models_to_train:
    train_model(x)