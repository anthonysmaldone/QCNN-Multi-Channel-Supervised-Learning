###########################
# build quantum convolutional neural network
def DR_U1_QCNN_model():
    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=12,activation='relu',
                      name='DR_U1_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=3,activation='relu',
                      name='DR_U1_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=1,activation='relu',
                      name='DR_U1_QCNN')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)
    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)
    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DR_U1_QCNN')
###########################
# build quantum convolutional neural network
def DRR_U1_QCNN_model():
    
    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=12,registers=2, rdpa=2, inter_U=True,activation='relu',
                      name='DRR_U1_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=3,registers=3, rdpa=1, inter_U=True,activation='relu',
                      name='DRR_U1_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U1_circuit(filter_size=2, n_kernels=3, n_input_channels=1,registers=1, rdpa=1, inter_U=True,activation='relu',
                      name='DRR_U1_QCNN')(x_input)
  
    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DRR_U1_QCNN')
############################
# build quantum convolutional neural network
def QCNN_U1_control_model():

    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U1_control(filter_size=2, n_kernels=3, activation='relu',
                      name='U1_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'Control_U1_QCNN')
############################
# build quantum convolutional neural network
def QCNN_U1_weighted_control_model():

    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U1_control(filter_size=2, n_kernels=3, classical_weights=True, activation='relu',
                      name='weighted_U1_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'MW_U1_QCNN')
###########################
# build quantum convolutional neural network
def DR_U2_QCNN_model():
    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=12,activation='relu',
                      name='DR_U2_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=3,activation='relu',
                      name='DR_U2_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=1,activation='relu',
                      name='DR_U2_QCNN')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DR_U2_QCNN')
###########################
# build quantum convolutional neural network
def DRR_U2_QCNN_model():
    if datatype == "CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=12,registers=2, rdpa=2, inter_U=True,activation='relu',
                      name='DRR_U2_QCNN')(x_input)

    if datatype == "COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=3,registers=3, rdpa=1, inter_U=True,activation='relu',
                      name='DRR_U2_QCNN')(x_input)

    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

        x_qconv1 = U2_circuit(filter_size=2, n_kernels=3, n_input_channels=1,registers=1, rdpa=1, inter_U=True,activation='relu',
                      name='DRR_U2_QCNN')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'DRR_U2_QCNN')
############################
# build quantum convolutional neural network
def QCNN_U2_control_model():
    
    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')



    x_qconv1 = Q_U2_control(filter_size=2, depth=3, activation='relu',
                      name='U2_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)
    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'Control_U2_QCNN')
############################
# build quantum convolutional neural network
def QCNN_U2_weighted_control_model():
    
    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')

    x_qconv1 = Q_U2_control(filter_size=2, depth=3, classical_weights=True, activation='relu',
                      name='weighted_U2_control')(x_input)

    x_flatten = tf.keras.layers.Flatten()(x_qconv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'MW_U2_QCNN')
############################
def CNN_classical_model():
    if datatype=="CHANNELS":
        x_input = tf.keras.layers.Input((10,10,12), name = 'input')
    if datatype=="COLORS" or datatype == "CIFAR10" or datatype == "COLORS_SHAPE":
        x_input = tf.keras.layers.Input((10,10,3), name = 'input')
    if datatype == "MNIST":
        x_input = tf.keras.layers.Input((10,10,1), name = 'input')
    
    x_conv1 = tf.keras.layers.Conv2D(3, (2, 2), activation='relu')(x_input)

    #x_conv2 = tf.keras.layers.Conv2D(5, (2, 2), activation='relu')(x_conv1)

    #x_conv3 = tf.keras.layers.Conv2D(5, (2, 2), activation='relu')(x_conv2)

    x_flatten = tf.keras.layers.Flatten()(x_conv1)

    x_fc1 = tf.keras.layers.Dense(32, activation='relu')(x_flatten)

    #x_dropout = tf.keras.layers.Dropout(global_dropout_rate,seed=42)(x_fc1)

    if datatype == "COLORS":
        x_fc2 = tf.keras.layers.Dense(8, activation='softmax')(x_fc1)
    
    elif datatype == "COLORS_SHAPE":
        x_fc2 = tf.keras.layers.Dense(16, activation='softmax')(x_fc1)

    else:
        x_fc2 = tf.keras.layers.Dense(10, activation='softmax')(x_fc1)

    return tf.keras.models.Model(inputs = x_input, outputs = x_fc2, name = 'CNNClassical')
############################