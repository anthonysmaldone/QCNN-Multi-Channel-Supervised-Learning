# import packages
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model
import numpy as np
import os

# choose dataset to train on
#   Options:
#   MNIST
#   COLORS
#   COLORS_SHAPES
#   CIFAR10
#   edit prepare_data.py file to change classes to train on for CIFAR-10
datatype = "CIFAR10"

# choose hyperparameters
num_of_epochs = 20
global_learning_rate = 0.01
global_batch_size = 50

# choose image size
resize_x = 10 
resize_y = 10 

# import project functions
from prepare_data import datasize, build_model_datasets
import generate_output
import models

# list containing train size, image size x, image size y, learning_rate, batch size, test size, dataset, number of epochs
details = [datasize(datatype)[0],resize_x,resize_y,global_learning_rate,global_batch_size,datasize(datatype)[1],datatype,num_of_epochs]

# set model to True to run
############################
DR_U1_QCNN = False 
DRR_U1_QCNN = False 
MW_U1_QCNN = False 
control_U1_QCNN = False 

DR_U2_QCNN = True
DRR_U2_QCNN = False 
MW_U2_QCNN = False
control_U2_QCNN = False
#############################
models_to_train = []


#############################
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
#############################



def train_model(model_to_train):
    model = model_to_train
##########################
# print the architecture of the model
    model.summary()
############################
# grab the time the training starts, output folder will be named with this time
    timestr_ = time.strftime("%Y%m%d-%H%M%S")
# compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=global_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# preprocess the chosen dataset
    model_data = build_model_datasets(datatype,details)
# begin to train the model
    model_history = model.fit(model_data[0], model_data[2], validation_data=(model_data[1],model_data[3]) , epochs=num_of_epochs, batch_size=global_batch_size)
# create timestampped folder to save output
    os.mkdir('output/'+timestr_)
# create confusion matrix
    print("CONFUSION MATRIX")
    classes = model_data[4]
    y_pred = model.predict(model_data[1])
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.flatten()
    if datatype == "CHANNELS" or datatype == "CIFAR10":
        y_true = model_data[3].flatten()
    else:
        y_true = model_data[3].numpy().flatten()
    confusion_mtx = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(16, 16))
    plt.figure()
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('output/'+timestr_+'/'+timestr+'_confusion_matrix.png', bbox_inches='tight')

# create learning curves plot
    print("GENERATE LEARNING CURVES")
    generate_output.save_output_imgs(model,model_history,details,timestr_)



# train all chosen models
for x in models_to_train:
    train_model(x)
