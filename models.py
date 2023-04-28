# import packages
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from circuits import U1_circuit, Q_U1_control, U2_circuit, Q_U2_control

###########################
# build quantum convolutional neural network
def DR_U1_QCNN_model(datatype):
# conditionally set input layer and quantum layer depending on the dataset

    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=12,activation='relu', datatype=datatype,
                      name='DR_U1_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=3,activation='relu',datatype=datatype,
                      name='DR_U1_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=1,activation='relu',datatype=datatype,
                      name='DR_U1_QCNN')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)
    
    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DR_U1_QCNN')
###########################
def DRR_U1_QCNN_model(datatype):
    
    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=12,registers=2, rdpa=2, inter_U=True, activation=tf.keras.layers.Activation('relu'),datatype=datatype, name='DRR_U1_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=3,registers=3, rdpa=1, inter_U=True,activation=tf.keras.layers.Activation('relu'),datatype=datatype,name='DRR_U1_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=1,registers=1, rdpa=1, inter_U=True,activation=tf.keras.layers.Activation('relu'),datatype=datatype,name='DRR_U1_QCNN')(x_input)
  
    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)


    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DRR_U1_QCNN')
############################
def QCNN_U1_control_model(datatype):

    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U1_control(filter_size=2, n_kernels=3, activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='U1_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'Control_U1_QCNN')
############################
def QCNN_U1_weighted_control_model(datatype):

    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U1_control(filter_size=2, n_kernels=3, classical_weights=True, activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='weighted_U1_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'MW_U1_QCNN')
###########################
def DR_U2_QCNN_model(datatype):
    
    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=12,activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='DR_U2_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=3,activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='DR_U2_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=1,activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='DR_U2_QCNN')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(2, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DR_U2_QCNN')
###########################
def DRR_U2_QCNN_model(datatype):

    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=12,registers=2, rdpa=2, inter_U=True,activation=tf.keras.layers.Activation('relu'),datatype=datatype,name='DRR_U2_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=3,registers=3, rdpa=1, ancilla=3,inter_U=True,activation=tf.keras.layers.Activation('relu'),datatype=datatype,name='DRR_U2_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=1,registers=1, rdpa=1, inter_U=True,activation=tf.keras.layers.Activation('relu'),datatype=datatype,name='DRR_U2_QCNN')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)


    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DRR_U2_QCNN')
############################
def QCNN_U2_control_model(datatype):
    
    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U2_control(filter_size=2, depth=3, activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='U2_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)
    
    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'Control_U2_QCNN')
############################
def QCNN_U2_weighted_control_model(datatype):
    
    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U2_control(filter_size=2, depth=3, classical_weights=True, activation=tf.keras.layers.Activation('relu'),datatype=datatype,
                      name='weighted_U2_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.Activation('relu'))(x_flatten)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(9, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(24, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'MW_U2_QCNN')
############################
