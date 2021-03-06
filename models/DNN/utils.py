import tensorflow as tf
import numpy as np
import os
import pickle
import json
import sys

def tf_xavier_init(fan_in, fan_out, *, const=1.0, epsilon = 1e-3, dtype=np.float32):
    '''
        Xavier initialization
    '''
    k = const * np.sqrt(1.0 / (fan_in + fan_out + epsilon))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)

def onehot(x, nclasses):
    '''
        input :
            x - 1D int32 array
            nclasses - max index of one-hot encoding
        return :
            2D array with onehot encoded
    '''
    temp = np.zeros((len(x), nclasses))
    for i in range(len(x)):
        try:
            temp[i][int(x[i])]=1		
        except IndexError:
            print("value of x({}) should not exceed nclasses({})".format(x[i], nclasses))
    return temp

def linear(x, fan_in, fan_out, activation = 'sigmoid'):
    '''
        input:
            x - 1D tensor
            fan_in - num of input nodes
            fan_out - num of output nodes
            activation - select the activation function
        return :
            1D tensor
            linear regression according to activation function 
    '''
    w = tf.Variable(tf_xavier_init(fan_in, fan_out), dtype = tf.float32, name = 'weights') 
    b = tf.Variable(tf.zeros([fan_out]), dtype = tf.float32, name = 'biases') 
    temp_linear = tf.matmul(x, w) + b
    
    if activation == 'sigmoid':
        return tf.sigmoid(temp_linear)
    elif activation == 'relu' :
        return tf.nn.relu(temp_linear)
    else :
        return tf.nn.tanh(temp_linear)
    
def shuffle(dataset):
    '''
        input:
            dataset - dictionary with 'input', and 'output'

        return :
            return the shuffle dictionary wiht 'input', and 'output'
    '''
    temp_input = dataset['input']
    temp_output = dataset['output']
    shuffle = np.arange(0, len(temp_input))
    np.random.shuffle(shuffle)
    shuffle_input = []
    shuffle_output = []
    for i in range(len(temp_input)):
        shuffle_input.append(temp_input[shuffle[i]])
        shuffle_output.append(temp_output[shuffle[i]])
    shuffle_input = np.array(shuffle_input)
    shuffle_output = np.array(shuffle_output)
    
    return {'input' : shuffle_input, 'output' : shuffle_output}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def txt_store(content, path):
    f = open(path, 'w')
    json.dump(content, f) 
    f.close()

def pickle_load(path):
    f = open(path, 'rb')
    temp = pickle.load(f)
    f.close()
    return temp

def pickle_store(content, path):
    f = open(path, 'wb')
    pickle.dump(content, f)
    f.close()
