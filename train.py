# import packages
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model
import numpy as np
import os

if not os.path.exists('output'):
    os.makedirs('output')

print("~~~~~~~~~~~MENU~~~~~~~~~~~~~")
print("1: COLORS")
print("2: COLORS_SHAPE")
print("3: 12-CHANNELS")
print("4: CIFAR-10")
try:
    datamenu1 = int(input('Choose dataset to train on (enter number): '))
except:
    datamenu1 = 4

try:
    datamenu2 = float(input('Enter learning rate (default 0.001): '))
except:
    datamenu2 = 0.001

if datamenu2 == '':
    datamenu2 = 0.001

if datamenu1 == 4:
    try:
        datamenu3 = int(input('Enter number of CIFAR classes (default 10): '))
    except:
        datamenu3 = 10
    
    if datamenu3 == '':
        datamenu3 = 10
else:
    datamenu3 = 10
    
print("Select models to run (y/n): ")

model1 = input('CO-QCNN (U1): ')
model2 = input('PCO-QCNN (U1): ')
model3 = input('PCO-T-QCNN (U1): ')
model4 = input('WEV-QCNN (U1): ')
model5 = input('Control QCNN (U1): ')
model6 = input('CO-QCNN (U2): ')
model7 = input('PCO-QCNN (U2): ')
model8 = input('PCO-T-QCNN (U2): ')
model9 = input('WEV-QCNN (U2): ')
model10 = input('Control QCNN (U2): ')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# choose dataset to train on

if datamenu1 == 1: 
    datatype = "COLORS"
if datamenu1 == 2: 
    datatype = "COLORS_SHAPE"
if datamenu1 == 3: 
    datatype = "CHANNELS"
if datamenu1 == 4: 
    datatype = "CIFAR10"
    

# choose hyperparameters
num_of_epochs = 20
global_learning_rate = datamenu2
global_batch_size = 50

#classes of CIFAR-10 dataset
classes = datamenu3

# choose image size
resize_x = 10 
resize_y = 10 

# import project functions
from prepare_data import datasize, build_model_datasets
import generate_output
import models

# list containing train size, image size x, image size y, learning_rate, batch size, test size, dataset, number of epochs
details = [datasize(datatype,classes)[0],resize_x,resize_y,global_learning_rate,global_batch_size,datasize(datatype,classes)[1],datatype,num_of_epochs]

# set model to True to run
############################
if model1 == "y":
    CO_U1_QCNN = True 
else:
    CO_U1_QCNN = False 

if model2 == "y":
    PCO_U1_QCNN = True 
else:
    PCO_U1_QCNN = False 

if model3 == "y":
    PCO_T_U1_QCNN = True 
else:
    PCO_T_U1_QCNN = False 
    
if model4 == "y":
    WEV_U1_QCNN = True 
else:
    WEV_U1_QCNN = False 
    
if model5 == "y":
    control_U1_QCNN = True 
else:
    control_U1_QCNN = False 
    
if model6 == "y":
    CO_U2_QCNN = True 
else:
    CO_U2_QCNN = False 
    
if model7 == "y":
    PCO_U2_QCNN = True 
else:
    PCO_U2_QCNN = False 
    
if model8 == "y":
    PCO_T_U2_QCNN = True 
else:
    PCO_T_U2_QCNN = False 
    
if model9 == "y":
    WEV_U2_QCNN = True 
else:
    WEV_U2_QCNN = False 
    
if model10 == "y":
    control_U2_QCNN = True 
else:
    control_U2_QCNN = False 
        

#############################
models_to_train = []
#############################
if CO_U1_QCNN:
    models_to_train.append(models.CO_U1_QCNN_model(datatype,classes))
    
if PCO_U1_QCNN:
    models_to_train.append(models.PCO_U1_QCNN_model(datatype,classes,3))    

if PCO_T_U1_QCNN:
    models_to_train.append(models.PCO_U1_QCNN_model(datatype,classes,1))

if WEV_U1_QCNN:
    models_to_train.append(models.QCNN_U1_weighted_control_model(datatype,classes))

if control_U1_QCNN:
    models_to_train.append(models.QCNN_U1_control_model(datatype,classes))

if CO_U2_QCNN:
    models_to_train.append(models.CO_U2_QCNN_model(datatype,classes))

if PCO_U2_QCNN:
    models_to_train.append(models.PCO_U2_QCNN_model(datatype,classes,3))
    
if PCO_T_U2_QCNN:
    models_to_train.append(models.PCO_U2_QCNN_model(datatype,classes,1))

if WEV_U2_QCNN:
    models_to_train.append(models.QCNN_U2_weighted_control_model(datatype,classes))

if control_U2_QCNN:
    models_to_train.append(models.QCNN_U2_control_model(datatype,classes))        
#############################

def train_model(model_to_train):
    global classes
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
    model_data = build_model_datasets(datatype,details,classes)
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
