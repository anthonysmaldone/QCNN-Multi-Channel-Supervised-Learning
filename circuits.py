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
#######################
# define a keras layer class to contain the quantum convolutional layer
class U1_circuit(tf.keras.layers.Layer):

    # initialize class
    def __init__(self, filter_size, n_kernels, n_input_channels, registers=1, rdpa=1, classical_weights=False, inter_U=False, activation=None, name=None, kernel_regularizer=None, **kwargs):
        super(U1_circuit, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.n_kernels = n_kernels
        self.n_input_channels = n_input_channels
        self.registers = registers
        self.rdpa = rdpa
        self.ancilla = int(registers/rdpa)
        self.classical_weights = classical_weights
        self.inter_U = inter_U
        self.learning_params = []
        self.Q_circuit()
        self.activation = tf.keras.layers.Activation(activation)
        self.kernel_regularizer = kernel_regularizer

    # define function to return a new learnable parameter, save all parameters
    # in self.learning_params
    def get_new_param(self):

        # generate symbol for parameter
        new_param = sympy.symbols("p"+str(len(self.learning_params)))

        # append new parameter to learning_params
        self.learning_params.append(new_param)

        # return the parameter
        return new_param



    # define quantum circuit
    def Q_circuit(self):
        # define number of pixels
        n_pixels = self.n_input_channels*(self.filter_size**2)
        circuit_layers = -(-self.n_input_channels//self.registers)
        qubit_registers = [cirq.GridQubit.rect(1, self.ancilla, top=0)]
        for i in range(self.registers):
          qubit_registers.append(cirq.GridQubit.rect(1, self.filter_size**2, top=i+1))
        
        
        # initialize qubits in circuit

        input_params = [sympy.symbols('a%d' %i) for i in range(n_pixels)]
        # intitialize circuit
        self.circuit = cirq.Circuit()
    
        # define function to entangle the inputs with a gate that applies a controlled
        # power of an X gate
        def Q_new_entangle(self, source, target, qubits_tar, qubits_src):
          yield cirq.CZPowGate(exponent=self.get_new_param())(qubits_tar[target], qubits_src[source])

        def Q_unentangle(self, source, target, qubits_tar, qubits_src, param):
          yield cirq.CZPowGate(exponent=-1*self.learning_params[param-1])(qubits_tar[target], qubits_src[source])

        
        def Q_embed(self,layer_index, register_index,qubits):
          starting_parameter = (self.filter_size**2)*(register_index+(layer_index*self.registers))
          for i in range(len(qubits)):
            self.circuit.append(cirq.rx(np.pi*input_params[starting_parameter+i])(qubits[i]))

        def Q_entangle_intra_data(self,qubits):
          self.circuit.append(Q_new_entangle(self,1, 0, qubits, qubits))
          self.circuit.append(Q_new_entangle(self,2, 1, qubits, qubits))
          self.circuit.append(Q_new_entangle(self,3, 2, qubits, qubits))
          self.circuit.append(Q_new_entangle(self,0, 3, qubits, qubits))
        
        def Q_entangle_inter_data(self,qubits_all):
          if self.registers > 2:
            for i in range(1,len(qubits_all),1):
              if i != len(qubits_all) - 1:
                self.circuit.append(Q_new_entangle(self,0, 0, qubits_all[i+1], qubits_all[i]))
              else:
                self.circuit.append(Q_new_entangle(self,0, 0, qubits_all[1], qubits_all[i]))
            
          else:
            self.circuit.append(Q_new_entangle(self,0, 0, qubits_all[1], qubits_all[2]))

        def Q_deposit(self,qubits,ancilla):
          self.circuit.append(cirq.CZPowGate(exponent=self.get_new_param())(qubits[0], qubit_registers[0][ancilla]))

        def Q_unentangle_inter_data(self,layer_index, register_index, qubits_all):
        
          if self.registers > 2:
            starting_parameter = 3+4*register_index+layer_index*6*self.registers+self.registers + 1
            count = 0
            for i in range(len(qubits_all)-1,0,-1):
              if i != len(qubits_all)-1:
                self.circuit.append(Q_unentangle(self, 0 ,0, qubits_all[i+1], qubits_all[i], starting_parameter-count))
              else:
                self.circuit.append(Q_unentangle(self, 0 ,0, qubits_all[1], qubits_all[i], starting_parameter-count))
              count = count + 1
          else:
            starting_parameter = 8+11*layer_index+1
            self.circuit.append(Q_unentangle(self, 0 ,0, qubits_all[1], qubits_all[2], starting_parameter))

        def Q_unentangle_intra_data(self,layer_index, register_index, qubits):
          
          if self.registers > 2 and self.inter_U:
            starting_parameter = 3+4*register_index+layer_index*6*self.registers
          
          elif self.registers > 2 and self.inter_U == False:
            starting_parameter = 3+4*register_index+layer_index*5*self.registers

          elif self.registers == 2 and self.inter_U:
            starting_parameter = 3+4*register_index+layer_index*5*self.registers+layer_index
          
          elif self.registers == 2 and self.inter_U == False:
            starting_parameter = 3+4*register_index+layer_index*5*self.registers
          
          else:
            starting_parameter = 3+4*register_index+layer_index*5
          
          self.circuit.append(Q_unentangle(self,0 ,3, qubits, qubits, starting_parameter+1))
          self.circuit.append(Q_unentangle(self,3 ,2, qubits, qubits, starting_parameter))
          self.circuit.append(Q_unentangle(self,2 ,1, qubits, qubits, starting_parameter-1))
          self.circuit.append(Q_unentangle(self,1 ,0, qubits, qubits, starting_parameter-2))

        def Q_unembed(self,layer_index,register_index,qubits):
          starting_parameter = (self.filter_size**2)*(register_index+(layer_index*self.registers))
          for i in range(len(qubits)):
            self.circuit.append(cirq.rx(-np.pi*input_params[starting_parameter+i])(qubits[i]))

        def Q_ancilla_entangle(self,qubits):
          if self.ancilla > 2:
            for i in range(self.ancilla):
              if i != self.ancilla - 1:
                  self.circuit.append(Q_new_entangle(self,i, i+1, qubits, qubits))
              else:
                  self.circuit.append(Q_new_entangle(self,0, i, qubits, qubits))
          else:
            self.circuit.append(Q_new_entangle(self,0, 1, qubits, qubits))
        
        for i in range(self.ancilla):
           self.circuit.append(cirq.H(qubit_registers[0][i]))
        
        for j in range(circuit_layers):
          if j != circuit_layers-1:
            for i in range(self.registers):
              Q_embed(self,j,i,qubit_registers[i+1])
            
            for i in range(self.registers):
              Q_entangle_intra_data(self,qubit_registers[i+1])
            
            if self.registers > 1 and self.inter_U:
              Q_entangle_inter_data(self,qubit_registers)
            
            ancilla_count = 1
            for i in range(self.registers):
              Q_deposit(self,qubit_registers[i+1],ancilla_count-1)
              if ancilla_count < self.ancilla:
                ancilla_count = ancilla_count + 1
            
            if self.registers > 1 and self.inter_U:
              Q_unentangle_inter_data(self,j,i,qubit_registers)
            
            for i in range(self.registers):
              Q_unentangle_intra_data(self,j,i,qubit_registers[i+1])
            
            for i in range(self.registers):
              Q_unembed(self,j,i,qubit_registers[i+1])
          
          else:
            for i in range(self.registers):
              Q_embed(self,j,i,qubit_registers[i+1])
            
            for i in range(self.registers):
              Q_entangle_intra_data(self,qubit_registers[i+1])
            
            if self.registers > 1 and self.inter_U:
              Q_entangle_inter_data(self,qubit_registers)
            
            ancilla_count = 1
            for i in range(self.registers):
              Q_deposit(self,qubit_registers[i+1],ancilla_count-1)
              if ancilla_count < self.ancilla:
                ancilla_count = ancilla_count + 1
            
        if self.registers > 1 and self.ancilla > 1:
          Q_ancilla_entangle(self,qubit_registers[0])

        print("Circuit Depth: "+str(len(cirq.Circuit(self.circuit.all_operations()))))

        # create list of embedding and learnable parameters
        self.params = input_params + self.learning_params

        # perform measurements on first qubit
        self.measurement = cirq.X(qubit_registers[0][0])

    # define keras backend function for initializing kernel
    def build(self, input_shape):

        self.width = input_shape[1]
        self.height = input_shape[2]

        self.num_x = self.width - self.filter_size + 1
        self.num_y = self.height - self.filter_size + 1

        # initialize kernel of shape(n_kernels, n_input_learnable_params)
        self.kernel = self.add_weight(name="kernel",
                                      shape=[self.n_kernels, 1, len(self.learning_params)],
                                      initializer=tf.keras.initializers.glorot_normal(seed=42),
                                      regularizer=self.kernel_regularizer)
        
        if self.classical_weights:
          self.classical_w = self.add_weight(name="classical_weights", 
                                             shape=[self.num_x, self.num_y],
                                             initializer=tf.keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                                             regularizer=self.kernel_regularizer)

        # create circuit tensor containing values for each convolution step
        # kernel will step num_x*num_y times
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y)
    # define a function to return a tensor of expectation values for each stride
    # for each data point in the batch
    def get_expectations(self, input_data, controller, circuit_batch):

        # input size: [batch_size*n_strides*n_input_channels, filter_size*filter_size]
        # controller shape: [batch_size*n_strides*n_input_channels, filter_size*filter_size]

        # concatenate input data and controller hoirzontally so that format is
        # commensurate with that of self.params
        input_data = tf.concat([input_data, controller], 1)

        # get expectation value for each data point for each batch for a kernel
        output = tfq.layers.Expectation()(circuit_batch,
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.measurement)
        # reshape tensor of expectation values to
        # shape [batch_size, n_horizontal_strides, n_vertical_strides,n_input_channels] and return
        output = tf.reshape(output, shape=[-1, self.num_x, self.num_y])
        if self.classical_weights:
          output = tf.math.multiply(output,self.classical_w)
        return output

    # define keras backend function to stride kernel and collect data
    def call(self, inputs):

        # define dummy variable to check if we are collecting data for first step
        stack_set = None

        # stride and collect data from input image
        for i in range(self.num_x):
            for j in range(self.num_y):

                # collecting image data superimposed with kernel
                # size = [batch_size, output_height, output_width, n_input_channels]
                slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size, self.filter_size, -1])

                # reshape to [batch_size, n_strides, filter_size, filter_size, n_input_channels]
                slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size, self.filter_size, self.n_input_channels])

                # if this is first stride, define new variable
                if stack_set == None:
                    stack_set = slice_part

                # if not first stride, concatenate to data from past strides
                else:
                    stack_set = tf.concat([stack_set, slice_part], 1)

        # permute shape to [batch_size, n_strides,  n_input_channels, filter_size, filter_size]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])

        # reshape to [batch_size*n_strides,n_input_channels*filter_size*filter_size]
        # each column corresponds to kernel's view of image, rows are ordered
        stack_set = tf.reshape(stack_set, shape=[-1, self.n_input_channels*(self.filter_size**2)])

        # create new tensor by tiling circuit values for each data point in batch
        circuit_batch = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])

        # flatten circuit tensor
        circuit_batch = tf.reshape(circuit_batch, shape=[-1])

        # initialize list to hold expectation values
        outputs = []
        for i in range(self.n_kernels):

            # create new tensor by tiling kernel values for each stride for each
            # data point in the batch
            controller = tf.tile(self.kernel[i], [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])

            # append to a list the expectations for all input data in the batch,
            # outputs is of shape [batch_size, n_horizontal_strides, n_vertical_strides]
            outputs.append(self.get_expectations(stack_set, controller, circuit_batch))

        # stack the expectation values for each kernel
        # shape is [batch_size, n_horizontal_strides, n_vertical_strides, n_kernels]
        output_tensor = tf.stack(outputs, axis=3)

        # take arccos of expectation and divide by pi to un-embed
        # if values are less than -1 or greater than 1, make -1 or 1, respectively
        output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi

        # return the activated tensor of expectation values
        return self.activation(output_tensor)
        

#######################
# define a keras layer class to contain the quantum convolutional layer
class U2_circuit(tf.keras.layers.Layer):

    # initialize class
    def __init__(self, filter_size, n_kernels, n_input_channels, registers=1, rdpa=1, classical_weights=False, inter_U=False, activation=None, name=None, kernel_regularizer=None, **kwargs):
        super(U2_circuit, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.n_kernels = n_kernels
        self.n_input_channels = n_input_channels
        self.registers = registers
        #register depositions per ancilla
        self.rdpa = rdpa
        self.ancilla = int(registers/rdpa)
        self.classical_weights = classical_weights
        self.inter_U = inter_U
        self.learning_params = []
        self.Q_circuit()
        self.activation = tf.keras.layers.Activation(activation)
        self.kernel_regularizer = kernel_regularizer

    # define function to return a new learnable parameter, save all parameters
    # in self.learning_params
    def get_new_param(self):

        # generate symbol for parameter
        new_param = sympy.symbols("p"+str(len(self.learning_params)))

        # append new parameter to learning_params
        self.learning_params.append(new_param)

        # return the parameter
        return new_param



    # define quantum circuit
    def Q_circuit(self):
        # define number of pixels
        n_pixels = self.n_input_channels*(self.filter_size**2)
        circuit_layers = -(-self.n_input_channels//self.registers)
        qubit_registers = [cirq.GridQubit.rect(1, self.ancilla, top=0)]
        for i in range(self.registers):
          qubit_registers.append(cirq.GridQubit.rect(1, self.filter_size**2, top=i+1))

        # initialize qubits in circuit

        input_params = [sympy.symbols('a%d' %i) for i in range(n_pixels)]
        # intitialize circuit
        self.circuit = cirq.Circuit()

        # define function to entangle the inputs with a gate that applies a controlled
        # power of an X gate
        def Q_new_entangle(self, source, target, qubits_tar, qubits_src, CZ=True):
          if CZ:
            yield cirq.CZPowGate(exponent=self.get_new_param())(qubits_tar[target], qubits_src[source])
          yield cirq.CXPowGate(exponent=self.get_new_param())(qubits_tar[target], qubits_src[source])

        def Q_unentangle(self, source, target, qubits_tar, qubits_src, param, CZ=True):
          yield cirq.CXPowGate(exponent=-1*self.learning_params[param])(qubits_tar[target], qubits_src[source])
          if CZ:
            yield cirq.CZPowGate(exponent=-1*self.learning_params[param-1])(qubits_tar[target], qubits_src[source])

        def Q_embed(self,layer_index, register_index,qubits):
          starting_parameter = (self.filter_size**2)*(register_index+(layer_index*self.registers))
          for i in range(len(qubits)):
            self.circuit.append(cirq.rx(np.pi*input_params[starting_parameter+i])(qubits[i]))

        def Q_entangle_intra_data(self,qubits):
          self.circuit.append(Q_new_entangle(self,1, 0, qubits, qubits))
          self.circuit.append(Q_new_entangle(self,3, 2, qubits, qubits))
          self.circuit.append(Q_new_entangle(self,2, 0, qubits, qubits))
        
        def Q_entangle_inter_data(self,qubits_all):
          if self.registers > 2:
            for i in range(1,len(qubits_all),1):
              if i != len(qubits_all) - 1:
                self.circuit.append(Q_new_entangle(self,0, 0, qubits_all[i+1], qubits_all[i], CZ=False))
              else:
                self.circuit.append(Q_new_entangle(self,0, 0, qubits_all[1], qubits_all[i], CZ=False))
          else:
            self.circuit.append(Q_new_entangle(self,0, 0, qubits_all[1], qubits_all[2], CZ=False))

        def Q_deposit(self,qubits,ancilla):
          self.circuit.append(cirq.CZPowGate(exponent=self.get_new_param())(qubits[0], qubit_registers[0][ancilla]))

        def Q_unentangle_inter_data(self,layer_index, register_index, qubits_all):
          if self.registers > 2:
            starting_parameter = 5+6*register_index+layer_index*8*self.registers+self.registers

            count = 0
            for i in range(len(qubits_all)-1,0,-1):
              if i != len(qubits_all)-1:
                self.circuit.append(Q_unentangle(self, 0 ,0, qubits_all[i+1], qubits_all[i], starting_parameter-count, CZ=False))
              else:
                self.circuit.append(Q_unentangle(self, 0 ,0, qubits_all[1], qubits_all[i], starting_parameter-count, CZ=False))
              count = count + 1
          else:
            starting_parameter = 12+layer_index*15
            self.circuit.append(Q_unentangle(self, 0 ,0, qubits_all[1], qubits_all[2], starting_parameter, CZ=False))
        
        def Q_unentangle_intra_data(self,layer_index, register_index, qubits):
          if self.registers > 2 and self.inter_U:
            starting_parameter = 5+6*register_index+layer_index*8*self.registers
          
          elif self.registers > 2 and self.inter_U == False:
            starting_parameter = 5+6*register_index+layer_index*7*self.registers
                   
          elif self.registers == 2 and self.inter_U:
            starting_parameter = 5+6*register_index+layer_index*7*self.registers + layer_index
          
          elif self.registers == 2 and self.inter_U == False:
            starting_parameter = 5+6*register_index+layer_index*7*self.registers
          
          else:
            starting_parameter = 5+6*register_index+layer_index*7
          
          self.circuit.append(Q_unentangle(self,2 ,0, qubits, qubits, starting_parameter))
          self.circuit.append(Q_unentangle(self,3 ,2, qubits, qubits, starting_parameter-2))
          self.circuit.append(Q_unentangle(self,1 ,0, qubits, qubits, starting_parameter-4))

        def Q_unembed(self,layer_index,register_index,qubits):
          starting_parameter = (self.filter_size**2)*(register_index+(layer_index*self.registers))
          for i in range(len(qubits)):
            self.circuit.append(cirq.rx(-np.pi*input_params[starting_parameter+i])(qubits[i]))
        
        def Q_ancilla_entangle(self,qubits):
          if self.ancilla > 2:
            for i in range(self.ancilla):
              if i != self.ancilla - 1:
                  self.circuit.append(Q_new_entangle(self,i, i+1, qubits, qubits,CZ=False))
              else:
                  self.circuit.append(Q_new_entangle(self,0, i, qubits, qubits,CZ=False))
          else:
            self.circuit.append(Q_new_entangle(self,0, 1, qubits, qubits,CZ=False))

        for i in range(self.ancilla):
           self.circuit.append(cirq.H(qubit_registers[0][i]))

        for j in range(circuit_layers):
          if j != circuit_layers-1:
            for i in range(self.registers):
              Q_embed(self,j,i,qubit_registers[i+1])
            
            for i in range(self.registers):
              Q_entangle_intra_data(self,qubit_registers[i+1])
            
            if self.registers > 1 and self.inter_U:
              Q_entangle_inter_data(self,qubit_registers)
            
            ancilla_count = 1
            for i in range(self.registers):
              Q_deposit(self,qubit_registers[i+1],ancilla_count-1)
              if ancilla_count < self.ancilla:
                ancilla_count = ancilla_count + 1
            
            if self.registers > 1 and self.inter_U:
              Q_unentangle_inter_data(self,j,i,qubit_registers)
            
            for i in range(self.registers):
              Q_unentangle_intra_data(self,j,i,qubit_registers[i+1])
            
            for i in range(self.registers):
              Q_unembed(self,j,i,qubit_registers[i+1])
          
          else:
            for i in range(self.registers):
              Q_embed(self,j,i,qubit_registers[i+1])
            
            for i in range(self.registers):
              Q_entangle_intra_data(self,qubit_registers[i+1])
            
            if self.registers > 1 and self.inter_U:
              Q_entangle_inter_data(self,qubit_registers)
            
            ancilla_count = 1
            for i in range(self.registers):
              Q_deposit(self,qubit_registers[i+1],ancilla_count-1)
              if ancilla_count < self.ancilla:
                ancilla_count = ancilla_count + 1
            
        if self.registers > 1 and self.ancilla > 1:
          Q_ancilla_entangle(self,qubit_registers[0])

        print("Circuit Depth: "+str(len(cirq.Circuit(self.circuit.all_operations()))))

        # create list of embedding and learnable parameters
        self.params = input_params + self.learning_params

        # perform measurements on first qubit
        self.measurement = cirq.X(qubit_registers[0][0])

    # define keras backend function for initializing kernel
    def build(self, input_shape):

        self.width = input_shape[1]
        self.height = input_shape[2]

        self.num_x = self.width - self.filter_size + 1
        self.num_y = self.height - self.filter_size + 1

        # initialize kernel of shape(n_kernels, n_input_learnable_params)
        self.kernel = self.add_weight(name="kernel",
                                      shape=[self.n_kernels, 1, len(self.learning_params)],
                                      initializer=tf.keras.initializers.glorot_normal(seed=42),
                                      regularizer=self.kernel_regularizer)
        
        if self.classical_weights:
          self.classical_w = self.add_weight(name="classical_weights", 
                                             shape=[self.num_x, self.num_y],
                                             initializer=tf.keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                                             regularizer=self.kernel_regularizer)

        # create circuit tensor containing values for each convolution step
        # kernel will step num_x*num_y times
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y)
    # define a function to return a tensor of expectation values for each stride
    # for each data point in the batch
    def get_expectations(self, input_data, controller, circuit_batch):

        # input size: [batch_size*n_strides*n_input_channels, filter_size*filter_size]
        # controller shape: [batch_size*n_strides*n_input_channels, filter_size*filter_size]

        # concatenate input data and controller hoirzontally so that format is
        # commensurate with that of self.params
        input_data = tf.concat([input_data, controller], 1)

        # get expectation value for each data point for each batch for a kernel
        output = tfq.layers.Expectation()(circuit_batch,
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.measurement)
        # reshape tensor of expectation values to
        # shape [batch_size, n_horizontal_strides, n_vertical_strides,n_input_channels] and return
        output = tf.reshape(output, shape=[-1, self.num_x, self.num_y])
        if self.classical_weights:
          output = tf.math.multiply(output,self.classical_w)
        return output

    # define keras backend function to stride kernel and collect data
    def call(self, inputs):

        # define dummy variable to check if we are collecting data for first step
        stack_set = None

        # stride and collect data from input image
        for i in range(self.num_x):
            for j in range(self.num_y):

                # collecting image data superimposed with kernel
                # size = [batch_size, output_height, output_width, n_input_channels]
                slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size, self.filter_size, -1])

                # reshape to [batch_size, n_strides, filter_size, filter_size, n_input_channels]
                slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size, self.filter_size, self.n_input_channels])

                # if this is first stride, define new variable
                if stack_set == None:
                    stack_set = slice_part

                # if not first stride, concatenate to data from past strides
                else:
                    stack_set = tf.concat([stack_set, slice_part], 1)

        # permute shape to [batch_size, n_strides,  n_input_channels, filter_size, filter_size]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])

        # reshape to [batch_size*n_strides,n_input_channels*filter_size*filter_size]
        # each column corresponds to kernel's view of image, rows are ordered
        stack_set = tf.reshape(stack_set, shape=[-1, self.n_input_channels*(self.filter_size**2)])

        # create new tensor by tiling circuit values for each data point in batch
        circuit_batch = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])

        # flatten circuit tensor
        circuit_batch = tf.reshape(circuit_batch, shape=[-1])

        # initialize list to hold expectation values
        outputs = []
        for i in range(self.n_kernels):

            # create new tensor by tiling kernel values for each stride for each
            # data point in the batch
            controller = tf.tile(self.kernel[i], [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])

            # append to a list the expectations for all input data in the batch,
            # outputs is of shape [batch_size, n_horizontal_strides, n_vertical_strides]
            outputs.append(self.get_expectations(stack_set, controller, circuit_batch))

        # stack the expectation values for each kernel
        # shape is [batch_size, n_horizontal_strides, n_vertical_strides, n_kernels]
        output_tensor = tf.stack(outputs, axis=3)

        # take arccos of expectation and divide by pi to un-embed
        # if values are less than -1 or greater than 1, make -1 or 1, respectively
        output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi
        
        #print()

        # return the activated tensor of expectation values
        return self.activation(output_tensor)
        
#######################
class Q_U1_control(tf.keras.layers.Layer):

    # initialize class
    def __init__(self, filter_size, n_kernels, padding=False, classical_weights=False, activation=None, name=None, kernel_regularizer=None, **kwargs):
        super(Q_U1_control, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.n_kernels = n_kernels
        self.classical_weights = classical_weights
        self.learning_params = []
        self.Q_circuit()
        self.activation = tf.keras.layers.Activation(activation)
        self.kernel_regularizer = kernel_regularizer

    # define function to return a new learnable parameter, save all parameters
    # in self.learning_params
    def get_new_param(self):

        # generate symbol for parameter
        new_param = sympy.symbols("p"+str(len(self.learning_params)))

        # append new parameter to learning_params
        self.learning_params.append(new_param)

        # return the parameter
        return new_param

    # define function to entangle the inputs with a gate that applies a controlled
    # power of an X gate
    def Q_entangle(self, source, target, qubits):
        yield cirq.CXPowGate(exponent=self.get_new_param())(qubits[source], qubits[target])

    # define quantum circuit
    def Q_circuit(self):
        # define number of pixels
        n_pixels = self.filter_size**2

        # initialize qubits in circuit
        cirq_qubits = cirq.GridQubit.rect(n_pixels,1)

        # intitialize circuit
        self.circuit = cirq.Circuit()

        # arbitrarily generate a symbol for each qubit
        input_params = [sympy.symbols('a%d' %i) for i in range(n_pixels)]

        # EMBED: tag each qubit with placeholder parameter, feed each to Pauli X gate
        for i, qubit in enumerate(cirq_qubits):
            self.circuit.append(cirq.rx(np.pi*input_params[i])(qubit))

        # ENTANGLE: strongly entangle all qubits
        for i in range(n_pixels):
            if i != n_pixels - 1:
              self.circuit.append(self.Q_entangle(i, i+1, cirq_qubits))
            else:
              self.circuit.append(self.Q_entangle(0, i, cirq_qubits))

        # create list of embedding and learnable parameters
        self.params = input_params + self.learning_params

        # perform measurements on first qubit
        
        self.measurement = cirq.Z(cirq_qubits[0])
        
        # define keras backend function for initializing kernel
    def build(self, input_shape):

        self.width = input_shape[1]
        self.height = input_shape[2]
        self.n_input_channels = input_shape[3]

        # define output dimensions for stride 1 with padding 1

        self.num_x = self.width - self.filter_size + 1
        self.num_y = self.height - self.filter_size + 1

        # initialize kernel of shape(n_kernels, n_input_channels, n_input_learnable_params
        self.kernel = self.add_weight(name="kernel",
                                      shape=[self.n_kernels, self.n_input_channels, len(self.learning_params)],
                                      initializer=tf.keras.initializers.glorot_normal(seed=42),
                                      regularizer=self.kernel_regularizer)

        if self.classical_weights:
            self.channel_weights = self.add_weight(name="channel_w",
                                          shape=[self.num_x,self.num_y,self.n_input_channels],
                                          initializer=tf.keras.initializers.Ones(),
                                          #initializer=tf.keras.initializers.RandomNormal(mean=1.0,stddev=0.1,seed=42),
                                          regularizer=self.kernel_regularizer)
        
            self.channel_bias = self.add_weight(name="channel_b",
                                          shape=[self.num_x,self.num_y,self.n_input_channels],
                                          initializer=tf.keras.initializers.Zeros(),
                                          #initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1,seed=42),
                                          regularizer=self.kernel_regularizer)



        # create circuit tensor containing values for each convolution step
        # kernel will step num_x*num_y*n_input_channels times
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y * self.n_input_channels)

    # define a function to return a tensor of expectation values for each stride
    # for each data point in the batch
    def get_expectations(self, input_data, controller, circuit_batch):

        # input size: [batch_size*n_strides*n_input_channels, filter_size*filter_size]
        # controller shape: [batch_size*n_strides*n_input_channels, filter_size*filter_size]

        # concatenate input data and controller hoirzontally so that format is
        # commensurate with that of self.params
        input_data = tf.concat([input_data, controller], 1)

        # get expectation value for each data point for each batch for a kernel
        output = tfq.layers.Expectation()(circuit_batch,
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.measurement)
        # reshape tensor of expectation values to
        # shape [batch_size, n_horizontal_strides, n_vertical_strides,n_input_channels] and return
        output = tf.reshape(output, shape=[-1, self.num_x, self.num_y, self.n_input_channels])
        if self.classical_weights:
            output = tf.math.multiply(output,self.channel_weights)
            output = tf.math.add(output,self.channel_bias)
        return tf.math.reduce_sum(output, 3)
    def call(self, inputs):

        # define dummy variable to check if we are collecting data for first step
        stack_set = None

        # stride and collect data from input image
        for i in range(self.num_x):
            for j in range(self.num_y):

                # collecting image data superimposed with kernel
                # size = [batch_size, output_height, output_width, n_input_channels]
                slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size, self.filter_size, -1])

                # reshape to [batch_size, n_strides, filter_size, filter_size, n_input_channels]
                slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size, self.filter_size, self.n_input_channels])

                # if this is first stride, define new variable
                if stack_set == None:
                    stack_set = slice_part

                # if not first stride, concatenate to data from past strides
                else:
                    stack_set = tf.concat([stack_set, slice_part], 1)

        # permute shape to [batch_size, n_strides,  n_input_channels, filter_size, filter_size]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])

        # reshape to [batch_size*n_strides*n_input_channels, filter_size*filter_size]
        # each column corresponds to kernel's view of image, rows are ordered
        stack_set = tf.reshape(stack_set, shape=[-1, self.filter_size**2])

        # create new tensor by tiling circuit values for each data point in batch
        circuit_batch = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])

        # flatten circuit tensor
        circuit_batch = tf.reshape(circuit_batch, shape=[-1])


        # initialize list to hold expectation values
        outputs = []
        for i in range(self.n_kernels):

            # create new tensor by tiling kernel values for each stride for each
            # data point in the batch
            controller = tf.tile(self.kernel[i], [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])

            # append to a list the expectations for all input data in the batch,
            # outputs is of shape [batch_size, n_horizontal_strides, n_vertical_strides]
            outputs.append(self.get_expectations(stack_set, controller, circuit_batch))

        # stack the expectation values for each kernel
        # shape is [batch_size, n_horizontal_strides, n_vertical_strides, n_kernels]
        output_tensor = tf.stack(outputs, axis=3)

        # take arccos of expectation and divide by pi to un-embed
        # if values are less than -1 or greater than 1, make -1 or 1, respectively
        output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi

        # return the activated tensor of expectation values
        return self.activation(output_tensor)
###########################
class Q_U2_control(tf.keras.layers.Layer):
    def __init__(self, filter_size, depth, classical_weights=False,activation=None, name=None, kernel_regularizer=None, **kwangs):
        super(Q_U2_control, self).__init__(name=name, **kwangs)
        self.filter_size = filter_size
        self.depth = depth
        self.learning_params = []
        self.QCNN_layer_gen()
        self.classical_weights = classical_weights
        # self.circuit_tensor = tfq.convert_to_tensor([self.circuit])
        self.activation = tf.keras.layers.Activation(activation)
        self.kernel_regularizer = kernel_regularizer

    def _next_qubit_set(self, original_size, next_size, qubits):
        step = original_size // next_size
        qubit_list = []
        for i in range(0, original_size, step):
            for j in range(0, original_size, step):
                qubit_list.append(qubits[original_size*i + j])
        return qubit_list

    def _get_new_param(self):
        """
        return new learnable parameter
        all returned parameter saved in self.learning_params
        """
        new_param = sympy.symbols("p"+str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param

    def _QConv(self, step, target, qubits):
        """
        apply learnable gates each quantum convolutional layer level
        """
        yield cirq.CZPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
        yield cirq.CXPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])

    def QCNN_layer_gen(self):
        """
        make quantum convolutional layer in QConv layer
        """
        pixels = self.filter_size**2
        # filter size: 2^n only for this version!
        if np.log2(pixels) % 1 != 0:
            raise NotImplementedError("filter size: 2^n only available")
        #The number of qubits required is determined by the size of the filter
        cirq_qubits = cirq.GridQubit.rect(self.filter_size, self.filter_size)
        # mapping input data to circuit
        input_circuit = cirq.Circuit()
        input_params = [sympy.symbols('a%d' %i) for i in range(pixels)]
        for i, qubit in enumerate(cirq_qubits):
            input_circuit.append(cirq.rx(np.pi*input_params[i])(qubit))
        # apply learnable gate set to QCNN circuit
        QCNN_circuit = cirq.Circuit()
        step_size = [2**i for i in range(np.log2(pixels).astype(np.int32))]
        for step in step_size:
            for target in range(0, pixels, 2*step):
                QCNN_circuit.append(self._QConv(step, target, cirq_qubits))
        # merge the circuits
        full_circuit = cirq.Circuit()
        full_circuit.append(input_circuit)
        full_circuit.append(QCNN_circuit)
        self.circuit = full_circuit # save circuit to the QCNN layer obj.
        self.params = input_params + self.learning_params
        self.op = cirq.Z(cirq_qubits[0])
    def build(self, input_shape):
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.channel = input_shape[3]
        self.num_x = self.width - self.filter_size + 1 #Because stride=1 and padding=0 (?)
        self.num_y = self.height - self.filter_size + 1

        self.kernel = self.add_weight(name="kernel",
                                      shape=[self.depth,
                                             self.channel,
                                             len(self.learning_params)],
                                     initializer=tf.keras.initializers.glorot_normal(),
                                     regularizer=self.kernel_regularizer)
        if self.classical_weights:
          self.classical_w = self.add_weight(name="classical_weights",
                                        shape=[self.num_x,
                                              self.num_y,
                                              self.channel],
                                      #initializer=tf.keras.initializers.RandomNormal(mean=1.0,stddev=0.1,seed=42),
                                      initializer=tf.keras.initializers.glorot_normal(),
                                      #initializer=tf.keras.initializers.Ones(),
                                      regularizer=self.kernel_regularizer)

          self.classical_b = self.add_weight(name="classical_bias",
                                        shape=[self.num_x,
                                              self.num_y,
                                              self.channel],
                                      #initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1,seed=42),
                                      initializer=tf.keras.initializers.glorot_normal(),
                                      #initializer = tf.keras.initializers.Zeros(),
                                      regularizer=self.kernel_regularizer)

        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y * self.channel)

    def call(self, inputs):
        # input shape: [N, width, height, channel]
        # slide and collect data
        stack_set = None
        for i in range(self.num_x):
            for j in range(self.num_y):
                slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size, self.filter_size, -1])
                slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size, self.filter_size, self.channel])
                if stack_set == None:
                    stack_set = slice_part
                else:
                    stack_set = tf.concat([stack_set, slice_part], 1)
        # -> shape: [N, num_x*num_y, filter_size, filter_size, channel]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])
        # -> shape: [N, num_x*num_y, channel, filter_size, fiter_size]
        stack_set = tf.reshape(stack_set, shape=[-1, self.filter_size**2])
        # -> shape: [N*num_x*num_y*channel, filter_size^2]

        # total input citcuits: N * num_x * num_y * channel
        circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
        circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])
        tf.fill([tf.shape(inputs)[0]*self.num_x*self.num_y, 1], 1)
        outputs = []
        for i in range(self.depth):
            controller = tf.tile(self.kernel[i], [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])
            outputs.append(self.single_depth_QCNN(stack_set, controller, circuit_inputs))
            # shape: [N, num_x, num_y]

        output_tensor = tf.stack(outputs, axis=3)
        output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi
        # output_tensor = tf.clip_by_value(tf.math.acos(output_tensor)/np.pi, -1, 1)
        return self.activation(output_tensor)
    def single_depth_QCNN(self, input_data, controller, circuit_inputs):
        """
        make QCNN for 1 channel only
        """
        # input shape: [N*num_x*num_y*channel, filter_size^2]
        # controller shape: [N*num_x*num_y*channel, len(learning_params)]
        input_data = tf.concat([input_data, controller], 1)
        # input_data shape: [N*num_x*num_y*channel, len(learning_params)]
        QCNN_output = tfq.layers.Expectation()(circuit_inputs,
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.op)
        # QCNN_output shape: [N*num_x*num_y*channel]
        #the -1 in the shape[] arg means the value is inferred from the length of the array and remaining dimensions
        QCNN_output = tf.reshape(QCNN_output, shape=[-1, self.num_x, self.num_y, self.channel])
        if self.classical_weights:
          QCNN_output = tf.math.multiply(QCNN_output,self.classical_w)
          QCNN_output = tf.math.add(QCNN_output,self.classical_b)
        return tf.math.reduce_sum(QCNN_output, 3)