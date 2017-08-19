import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle

def onset_times(sound, sampling_rate):
    '''
        input :
            sound - 1D array song
            sampling_rate - sampling_rate
        return : 
            1D array onset sequences (seconds)
    '''
    return librosa.frames_to_time(librosa.onset.onset_detect(y=sound, sr=sampling_rate), sr=sampling_rate)

def time_axies(array, sampling_rate):
    '''
        input :
            array- 1D array
            sampling_rate- int
        return :
            make suitable time axis with sampling_rate fiting well with array
    '''
    return np.linspace(0,(len(array)-1)/sampling_rate,len(array))

def onset_plot(sound, sampling_rate):
    '''
        input :
            sound - 1D array
            sampling_rate - int
        return :
            draw the sound plot with onset times is indicated
    '''
    onset_time = onset_times(sound, sampling_rate)
    time_axis = time_axies(sound, sampling_rate)
    plt.plot(time_axis, sound)
    plt.vlines(onset_time, np.min(sound), np.max(sound), color='r', alpha=0.7,
                    linestyle='--', label='Onsets')   
    plt.show()
    return

def DTFS(sound, islog = False, compressed_ratio = 100):
    '''
        input :
            
            sound : 1D array
            islog : boolean
            compressed_ratio : int

        return :
            perform DTFS(Discrete time fourier series)
            if  islog == True : normalize(log(1 + compressed_ratio*DTFS))
            else : normalize(DTFS)
    '''
    period = len(sound)
    a = np.zeros(period, dtype = np.complex64)
    for k in range(int(period/2)):
        for n in range(period):
            a[k] += sound[n]*np.exp(-1j*2*np.pi/period*k*n)
        a[k]/= period
    
    temp = np.array(abs(a), np.float32)
    if islog:
        return normalize(np.log(1 + compressed_ratio*temp))
    else :
        return normalize(temp)

def clip_by_value(x, v_max = 1, v_min = 0):
    if x>v_max:
        return v_max
    if x<v_min :
        return v_min

    return x

def pickle_load(path):
    f = open(path, 'rb')
    temp = pickle.load(f)
    f.close()
    return temp

def pickle_store(content, path):
    f = open(path, 'wb')
    pickle.dump(content, f)
    f.close()

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
        input :
            
            sound : 1D array
            islog : boolean
            compressed_ratio : int

        return :
            perform DTFS(Discrete time fourier series)
            if  islog == True : normalize(log(1 + compressed_ratio*DTFS))
            else : normalize(DTFS)
    '''
    period = len(sound)
    a = np.zeros(period, dtype = np.complex64)
    for k in range(int(period/2)):
        for n in range(period):
            a[k] += sound[n]*np.exp(-1j*2*np.pi/period*k*n)
        a[k]/= period
    
    temp = np.array(abs(a), np.float32)
    if islog:
        return normalize(np.log(1 + compressed_ratio*temp))
    else :
        return normalize(temp)

