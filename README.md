# Sound classifier code
based on the **onset detection** + **Neural Network**

## Environments
    tensorflow 1.0.1
    numpy 1.12.0
    matplotlib
    librosa 0.4.3
  
## Datas source
[Drum bit collection](http://soundpacks.com/free-sound-packs/)

## Data arrangement
* Locate '../directory/' **directory** : current setting(=**sound_datas2**)
* In the **directory**, several types of **sound_container**(=../sound_datas2/clap, bass, hihat, snare, kick, percussion)
* Each sound name with style **kick001.wav** is stored in each **sound_container**

## Load the data
### Run
Run **dataloader.py** in the **preprocess** directory
* Implement
~~~
    dataloader.py --forward 0.03 --backward 0.07 --islog True --comp 1.0** 
~~~
* Then, clip the sound from (first onset - 0.03 ~ first onset + 0.07)
* Use the **DTFS** to process the clipped sounds to make input data
* Transform the spectogram with 
> if islog == True => **log(1 + comp*abs(spectogram))**  
> if islog == False => **abs(spectogramgram)**  
*  Make **./dataset/f0.03b0.07logTruecomp1.0.pkl** with pickle which contains 'input data', 'output data', 'sound_type'
> **input data** : DTFS of clipped sounds     
> **output data** : sound class label corresponding to input data   
> **sound_type** : explainsthe class of output data with dictionary  
> *ex) {0: 'clap', 1: 'snare', 2: 'percussion', 3: 'kick', 4: 'bass', 5: 'hihat'}*

## Train the model
* models are located in './models/'  
* Currently there are 2 models.  
* Exclude the label seems to be weird with **exclude_label**  

### DNN
In './models/DNN/',  
~~~
    DNN.py  
~~~

* Reads the corresponding dataset file such as **./dataset/f0.03b0.07logTruecomp1.0.pkl**  
* Train the model with the dataset  
* Save the **trained model** in **./models_save/DNN/f0.03b0.07logTruecomp1.0/** directory  
* Save the **info.pkl, info.txt** in **./models_save/DNN/f0.03b0.07logTruecomp1.0/** directory  
* Both are same, but **info.txt** => human readable while **info.pkl** => for usage  
> **exlude_label** : exclude label in training data  
> **sound_type** : explanation of sound_type    

### DNN_dropconnect
In './models/DNN_dropconnect/',   
~~~
    DNN_dropconnect.py
~~~
* From DNN, added the concept of dropconnect.  

## onset_classifier.py 

### Model_available
Changes the model with model_name

1. DNN
2. DNN_dropconnect

### Usage
~~~
    classifier = OnsetClassifier(model = , # model defaults to be 'DNN'
                                 forward = , # forward defaults to be 0.03
                                 backward = , # backward defaults to be 0.07
                                 islog = , # islog defaults to be True
                                 comp = ) # comp defaults to be 1.0 

    features = feature_extractor(y, sr)
    # while y : sound data 
    # sr : sampling rate
    classifier.sound_type # sound_type shows how the labels looks like 
    classifier.exclude_label # exclude_lable gives info related to excluded labed while training
~~~

## Test the model
**tester.ipynb** to use onset_classifier.py
Test the **?.wav** with **?.png** controlled by duration
> The original sound is on the below  
> For each legend, the graph explains how the onset looks like.  
> Store the **.png** image on **asset** directory  
