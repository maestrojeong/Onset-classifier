import tensorflow as tf
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)

def conv1d(x, f, hop_length = 1, padding = 'SAME', scope = "conv1d"): 
    with tf.name_scope(scope):
        batch, width, in_channel = get_shape(x)
        window, in_channel, out_channel = get_shape(f)

        x_r = tf.reshape(x, [batch, 1, width, in_channel])
        f_r = tf.reshape(f, [1, window, in_channel, out_channel]) 

        conv = tf.nn.conv2d(input= x_r, filter = f_r
                                , strides=[1, hop_length, 1, 1], padding = padding)
        conv_r = tf.reshape(conv, [batch, -1, out_channel], name = 'output')
        
        logger.debug('[conv1d] %s : %s(%s)->%s(%s)'\
                % (scope, x.name, get_shape(x), conv_r.name, get_shape(conv_r)))
        return conv_r
        
def get_shape(x):
    return x.get_shape().as_list()

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

def conv1D(x, filter_size, scope = ):
    '''
        Implement conv1D using conv2D
        input :

        return :

    '''
    
    w = weights

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

def pickle_load(path):
    f = open(path, 'rb')
    temp = pickle.load(f)
    f.close()
    return temp

def pickle_store(content, path):
    f = open(path, 'wb')
    pickle.dump(content, f)
    f.close()
