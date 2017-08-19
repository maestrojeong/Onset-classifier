import os
import pickle
import numpy as np
import librosa

def number_clean(number):
    """
        number should be integer
        1->001
        10->010
        otherwise same
    """
    if number < 10:
        return '00' + str(number)
    if number < 100:
        return '0' + str(number)
    else:
        return str(number)

def music_loader(sound_dir, sound_type, sequence, sound_type_dict, print_option = False):
    '''
        input:
            sound_dir : string(indicates the sound data directory)
            sound_type : int(key of type in sound_type_dict)    
            sequene : song_number
            sound_type_dict - dictionary 
            print_option-bool
                true => print info 
                false => nothing
        return : 
            'y' : music sample
            'sr' : sampling_rate
    '''
    temp_dir = os.path.join(sound_dir, sound_type_dict[sound_type])
    music_name = "{}.wav".format(sound_type_dict[sound_type]+number_clean(sequence+1))
    # Starts from 1
    music_path = os.path.join(temp_dir, music_name)
    a, b = librosa.load(music_path)
    if print_option:
        print("{} is loaded".format(music_path))

    return {'y' : a, 'sr' : b}  #  a : y, b : sampling_rate 

def onset_times(sound, sampling_rate):
    '''
        input :
            sound - 1D array song
            sampling_rate - sampling_rate
        return : 
            1D array onset sequences (seconds)
    '''
    return librosa.frames_to_time(librosa.onset.onset_detect(y=sound, sr=sampling_rate), sr=sampling_rate)

def zero_padding(y):
    '''
        input : 
            y - 1D array
        return :
            1D array with zero padding before
    '''
    temp = np.zeros(len(y))
    temp = np.append(temp, y)
    return temp

def normalize(x):
    '''
        input : 
            x - numpy 1D array
        return :
            1D array normalized to be 1
    '''
    sum_ = np.sum(x)
    return x/sum_

def DTFS(sound, islog = False, compressed_ratio = 100):
    '''
    Arg :
        sound : 1D array
        islog : boolean
        compressed_ratio : int

    Return :
        perform DTFS(Discrete time fourier series)
        if  islog == True : normalize(log(1 + compressed_ratio*DTFS))
        else : normalize(DTFS)
    '''
    period = len(sound)
    fourier = np.fft.fft(sound) # DTFT of sound

    # Get half of DTFT
    abs_fourier_half = np.zeros(period, dtype = np.float32) 
    for k in range(int(period/2)):
        abs_fourier_half[k] = abs(fourier[k])
    
    if islog:
        return normalize(np.log(1 + compressed_ratio*abs_fourier_half))
    else :
        return normalize(abs_fourier_half)

def clip_onset(sound, sampling_rate, forward_diff = 0.03, backward_diff = 0.07):
    '''
        input :
            
            sound : 1D array
            sampling_rate : int
            foward_diff : float default to be 0.03
            backward_diff : float default to be 0.05

        return 
            Clip around the first onset times
            (first onset times - forward_diff(seconds) ~ first onset times + backward_diff(seconds))
    '''
    onsets = onset_times(sound, sampling_rate)
    if len(onsets) > 0:
        standard = int(onsets[0]*sampling_rate)
        start = standard - int(forward_diff*sampling_rate)
        end = standard + int(backward_diff*sampling_rate)
    else:
        return None
    if start >=0 and end < len(sound):
        return sound[start:end] 
    else:
        return None

def pickle_store(content, path):
    f = open(path, 'wb')
    pickle.dump(content, f)
    f.close()
