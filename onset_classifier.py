from utils import pickle_load
from utils import clip_by_value
from utils import DTFS
from utils import onset_times
import tensorflow as tf
import numpy as np
import os

def onset_classifier(y, sr = 22050, model_name = 'DNN', forward = 0.03, backward = 0.07, islog = True, comp = 1.0):
    '''
    feature extraction with model

    Args
        y - 1D array
            mono sound_data
        sr - int(currently should be default setting of libtrosa 22050)
            sampling_rate of sound
        model_name - string 
            name of model
        forward - float
            clip from onset - forward
        backward - float
            clip to onset + backward
        islog - bool 
            log on the spectrum on DTFS
        comp - float 
            compression ratio
    return :
        exclude_label - exclude_label
        sound_type - dict
        features - 2D array(self.nclasses, len(y))
    '''

    MODEL_PATH = './models_save/{}/f{}b{}log{}comp{}/'.format(model_name, 
                                                              forward,
                                                              backward,
                                                              islog,
                                                              comp)

    info = pickle_load(os.path.join(MODEL_PATH, 'info.pkl'))
    exclude_label = info['exclude_label']
    sound_type = info['sound_type']

    nclasses = len(sound_type)
    sound_length = len(y)

    onsets = onset_times(y, sr)        
    n_onsets = len(onsets)
    
    sound_classify = np.zeros((len(onsets), nclasses))
    features = np.zeros((nclasses, len(y)))
    

    # sound_classify update 
    with tf.Session() as sess:
        # Neural network restoration
        restorer = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'model.meta'))
        restorer.restore(sess, os.path.join(MODEL_PATH, 'model'))
        freqs = tf.get_collection("input")[0]
        index = tf.get_collection("output")[0]

        if model_name == 'DNN_dropconnect':
            istrain = tf.get_collection("istrain")[0]

        # Apply neural network
        feed_dict = {}
        if model_name == 'DNN_dropconnect':
            feed_dict[istrain] = False

        for i in range(n_onsets):
            standard = int(onsets[i]*sr)
            clip_from = standard - int(forward*sr)
            clip_to = standard + int(backward*sr)
            if clip_from>=0 and clip_to<sound_length:
                dtfs = DTFS(sound = y[clip_from:clip_to],
                            islog = islog,
                            compressed_ratio = comp) 
                feed_dict[freqs] =  np.reshape(dtfs, [1,-1]) 
                temp = np.reshape(sess.run(index, feed_dict = feed_dict), [-1])
                for j in range(nclasses):
                    sound_classify[i][j] = temp[j]

    tf.reset_default_graph()

    # features update
    for i in range(len(onsets)):
        standard = int(onsets[i]*sr)
        clip_from = standard - int(forward*sr)
        clip_to = standard + int(backward*sr)
        if clip_from>=0 and clip_to<sound_length:
            for j in range(clip_from, clip_to):
                for k in range(nclasses):
                    features[k][j] += sound_classify[i][k]

    # clip the value of features from 0 to 1
    for i in range(nclasses):
        for j in range(sound_length):
            features[i][j] = clip_by_value(features[i][j], 1, 0)

    return exclude_label, sound_type, features
