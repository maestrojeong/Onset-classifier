from utils import number_clean
from utils import music_loader
from utils import pickle_store
from utils import zero_padding
from utils import clip_onset
from utils import DTFS
import os
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--forward", default = 0.03, help="clip forward difference(secs)", type = float)
parser.add_argument("--backward", default = 0.07, help="clip backward difference(secs)", type = float)
parser.add_argument("--islog", default = False, help = "Determine form of DTFS", type = bool)
parser.add_argument("--comp", default = 1.0, help = "Compressed ratio for DTFS", type = float)

args = parser.parse_args()

# Hyperparameters
## Parse the sounds from (first_onset - forward) to (first_onset + backward)
forward = args.forward
backward = args.backward
## Control the processing after DTFS
islog = args.islog
comp_ratio = args.comp

sound_dir = '../../sound_datas2'
sound_dir_keys = os.listdir(sound_dir) # List for sound directory names
nsoundtypes = len(sound_dir_keys) # Number of sound type
sound_type_dict = {} # { index : sound_directory names}  index <= 0 ~ nsoundtypes - 1
nsounds = [] # Number of sounds for each type
input_data = [] # DTFT result of parsed sound
output_data = [] # types of corresponding input_data 
store_file = '../dataset/f{}b{}log{}comp{}.pkl'.format(forward, backward, islog, comp_ratio) # Store file here

sound_dir_keys = sorted(sound_dir_keys)

for i in range(nsoundtypes):
    sound_type_dict[i] = sound_dir_keys[i]

for dir_key in sound_dir_keys:
    temp_dir = os.path.join(sound_dir, dir_key)
    nsounds.append(len(os.listdir(temp_dir)))

min_nsounds = min(nsounds) # minimum number of sounds

print(sound_type_dict)
print("Each sounds have {}".format(nsounds))
print("Min_sounds : {}".format(min_nsounds))

for i in range(nsoundtypes):
    for j in range(min_nsounds):
        try :
            result = music_loader(sound_dir, i,j, sound_type_dict, True)
            y = result['y']
            sr = result['sr']
            
            try : 
                clipped_sound = clip_onset(zero_padding(y), sr, forward_diff = forward, backward_diff = backward)
                if clipped_sound is None:
                    continue
            except IndexError:
                continue

            temp = DTFS(clipped_sound, islog = islog, compressed_ratio = comp_ratio)
            input_data.append(temp)
            output_data.append(i)

        except FileNotFoundError:
            print("File not found")

input_data = np.array(input_data)
output_data = np.array(output_data)

print("input data : {}".format(input_data.shape))
print("output data : {}".format(output_data.shape)) 

dataset = {'input' : input_data, 'output' : output_data, 'sound_type' : sound_type_dict}
pickle_store(dataset, store_file) 
