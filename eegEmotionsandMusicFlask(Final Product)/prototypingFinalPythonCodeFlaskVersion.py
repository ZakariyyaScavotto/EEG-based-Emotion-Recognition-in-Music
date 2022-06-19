from pylsl import StreamInlet, resolve_stream #Used to read in EEG data from EmotivPRO software
from joblib import load #used to load the SVMs
import scipy.signal as signal #used for bandpass filtering, Welch's method
from scipy.stats import differential_entropy #used to calculate DE for feature reduction
import numpy as np

#REMEMBER: BEFORE RUNNING MAKE SURE THAT EMOTIV LSL IS RUNNING

def mainBackground():
    # first resolve an EEG stream on the lab network
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    #0) load SVMs, initialize other thiings used repeatedly (i.e. the EEG filter)
    arousalSVM = load('arousalSVM14Channel.joblib')
    valenceSVM = load('valenceSVM14Channel.joblib')

    lowPassCutoff, highPassCutoff = 2/(0.5*128), 40/(0.5*128) #cutoff = cutoffFrequency(Hz)/(0.5*sample rate (emotiv records at sample rate of 128Hz))
    b, a = signal.butter(1, [lowPassCutoff, highPassCutoff], btype='bandpass',output='ba')
    while True:       
        #1) Get list of channels
        channels = [[] for i in range(14)]
        for sampleNum in range(2048): #DEAP had 8064 points, want to use the same length ideally
            #should this number be cut down because of how long it takes to get so many points?
            sample = inlet.pull_sample()
            for channelInd in range(3,17):
                channels[channelInd-3].append(sample[0][channelInd])

        #2) Bandpass filter
        for i in range(14):
            channels[i] = signal.filtfilt(b,a, channels[i])

        #3) get Welch of each channel
        for i in range(14):
            frequencies, channels[i] = signal.welch(channels[i], fs=128, nperseg=1024)
        
        #4) get the DE of each channel
        DEList = [differential_entropy(channels[i]) for i in range(14)]

        #5) feed into each SVM
        DEarray = [np.asarray(DEList)]
        arousalPredicts, valencePredicts = arousalSVM.predict(DEarray), valenceSVM.predict(DEarray)
        
        #6) get final emotion response
        arousal, valence = arousalPredicts[0], valencePredicts[0]
        if arousal == 1 and valence == 1: #1st quad of valence/arousal
            print('HAPPY') 
            return 'HAPPY'
        elif arousal == 1 and valence == -1: #2nd quad of valence/arousal
            print('WORRIED')
            return 'WORRIED'
        elif arousal == -1 and valence == 1: #4th quad of valence/arousal
            print('CALM')
            return 'CALM'
        else: #3rd quad of valence/arousal
            print('SAD')
            return 'SAD'
