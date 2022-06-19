from pylsl import StreamInlet, resolve_stream #Used to read in EEG data from EmotivPRO software
from joblib import load #used to load the SVMs
import scipy.signal as signal #used for bandpass filtering, Welch's method
import numpy as np
from statistics import mean
import math
import pywt

#REMEMBER: BEFORE RUNNING MAKE SURE THAT EMOTIV LSL IS RUNNING

def mainBackground():
    # first resolve an EEG stream on the lab network
    print('program starting')
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    #0) load SVMs, initialize other things used repeatedly (i.e. the EEG filter)
    # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    channel0ArousalSVM = load('bazgirSVMs/channel0ArousalSVM.joblib')
    channel1ArousalSVM = load('bazgirSVMs/channel1ArousalSVM.joblib')
    channel2ArousalSVM = load('bazgirSVMs/channel2ArousalSVM.joblib')
    channel3ArousalSVM = load('bazgirSVMs/channel3ArousalSVM.joblib')
    channel4ArousalSVM = load('bazgirSVMs/channel4ArousalSVM.joblib')
    channel5ArousalSVM = load('bazgirSVMs/channel5ArousalSVM.joblib')
    channel6ArousalSVM = load('bazgirSVMs/channel6ArousalSVM.joblib')
    channel7ArousalSVM = load('bazgirSVMs/channel7ArousalSVM.joblib')
    channel8ArousalSVM = load('bazgirSVMs/channel8ArousalSVM.joblib')
    channel9ArousalSVM = load('bazgirSVMs/channel9ArousalSVM.joblib')
    channel10ArousalSVM = load('bazgirSVMs/channel10ArousalSVM.joblib')
    channel11ArousalSVM = load('bazgirSVMs/channel11ArousalSVM.joblib')
    channel12ArousalSVM = load('bazgirSVMs/channel12ArousalSVM.joblib')
    channel13ArousalSVM = load('bazgirSVMs/channel13ArousalSVM.joblib')
    channel0ValenceSVM = load('bazgirSVMs/channel0ValenceSVM.joblib')
    channel1ValenceSVM = load('bazgirSVMs/channel1ValenceSVM.joblib')
    channel2ValenceSVM = load('bazgirSVMs/channel2ValenceSVM.joblib')
    channel3ValenceSVM = load('bazgirSVMs/channel3ValenceSVM.joblib')
    channel4ValenceSVM = load('bazgirSVMs/channel4ValenceSVM.joblib')
    channel5ValenceSVM = load('bazgirSVMs/channel5ValenceSVM.joblib')
    channel6ValenceSVM = load('bazgirSVMs/channel6ValenceSVM.joblib')
    channel7ValenceSVM = load('bazgirSVMs/channel7ValenceSVM.joblib')
    channel8ValenceSVM = load('bazgirSVMs/channel8ValenceSVM.joblib')
    channel9ValenceSVM = load('bazgirSVMs/channel9ValenceSVM.joblib')
    channel10ValenceSVM = load('bazgirSVMs/channel10ValenceSVM.joblib')
    channel11ValenceSVM = load('bazgirSVMs/channel11ValenceSVM.joblib')
    channel12ValenceSVM = load('bazgirSVMs/channel12ValenceSVM.joblib')
    channel13ValenceSVM = load('bazgirSVMs/channel13ValenceSVM.joblib')

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
        print('Data collected')
        #2) Bandpass filter
        for i in range(14):
            channels[i] = signal.filtfilt(b,a, channels[i])
        subjectNumber = 0
        subjectDataDict = {}
        subjectDataDict[subjectNumber] = [channel for channel in channels] 
        #3) average mean reference
        print('Starting mean referencing')
        newDataSubject = []
        trial = subjectDataDict[subjectNumber]
        trialList = []
        for channel in trial:
            changedChannelList = []
            meanOfChannel = mean(channel)
            for sample in channel:
                changedChannelList.append(sample-meanOfChannel)
            # normalize between 0 and 1
            minimum, maximum = min(changedChannelList), max(changedChannelList)
            scaledChannelList = []
            for sample in changedChannelList:
                scaledChannelList.append((sample-minimum)/(maximum-minimum))
            # npChangedChannelList = np.array(changedChannelList)
            # norm = np.linalg.norm(npChangedChannelList)
            # scaledChannelList = npChangedChannelList / norm
            trialList.append(scaledChannelList)
        newDataSubject.append(trialList)
        subjectDataDict[subjectNumber] = newDataSubject[0]
        # 4) window
        windowedSignalDict = {}
        subjectWindowedData = []
        twoSeconds = 128
        print('Starting windowing')
        trial = subjectDataDict[subjectNumber]
        trialList = []
        #2 second windows to start
        for channel in trial:
            startIndex, endIndex = 0, twoSeconds
            channelList = []
            while endIndex < 2048:
                if len(channel[startIndex:endIndex]) != 0:
                    channelList.append(channel[startIndex:endIndex])
                startIndex += twoSeconds//2
                endIndex += twoSeconds//2 # want 50% overlap
            if len(channel[startIndex:]) != 0:
                channelList.append(channel[startIndex:])
            trialList.append(channelList)
        subjectWindowedData.append(trialList)
        windowedSignalDict[subjectNumber] = subjectWindowedData
        # 5) decompose signals into frequency bands by db4 mother wavelet function
        print('Starting DWT computations')
        subjectDWT = []
        trial = windowedSignalDict[subjectNumber][0]
        trialList = []
        for channel in trial:
            channelList = []
            for window in channel:
                # OLD, DIDN'T KNOW ABOUT DECOMPOSITION LEVELS
                # #cA, cD both are of length 82
                # (cA, cD) = pywt.dwt(window, 'db4')
                # channelList.append(cA)
                # NEW, KNOW ABOUT DECOMPOSITION LEVELS
                if len(window) != 0:
                    coeffs = pywt.wavedec(window, 'db4', level=4)
                    channelList.append(coeffs)
            trialList.append(channelList)
        subjectDWT.append(trialList)
        # 6) Get Entropy (and energy) of every frequency band from every window
        print('Starting feature calculations')
        subjectFeatureList = []
        for trial in subjectDWT:
            trialList = []
            for channel in trial:
                channelList = []
                for window in channel:
                    windowList = []
                    #compute entropy, energy for each band
                    for band in window:
                        entropySum = 0
                        for coeff in band:
                            entropySum += (coeff**2) * math.log(coeff**2, 10)
                        entropySum = entropySum * -1
                        windowList.append(entropySum)
                        # energy
                        energySum = 0
                        for coeff in band:
                            energySum += (coeff**2) 
                        windowList.append(energySum)
                    channelList.append(windowList)
                trialList.append(channelList)
            subjectFeatureList.append(trialList)
        # print(len(subjectFeatureList))
        subjectFeatureList = subjectFeatureList[0]
        #7) predict
        # print(len(subjectFeatureList))
        # print(len(subjectFeatureList[0]))
        # random stuff go!
        for channel in subjectFeatureList:
            while len(channel) < 113:
                channel.append(channel[0])
        # newFeatureList = [item for item in window for window in channel for channel in subjectFeatureList]
        newFeatureList = []
        for channel in subjectFeatureList:
            newChannel = []
            for window in channel:
                for item in window:
                    newChannel.append(item)
            newFeatureList.append(newChannel)
        channel0Data, channel1Data, channel2Data, channel3Data, channel4Data, channel5Data, channel6Data, channel7Data, channel8Data, channel9Data, channel10Data, channel11Data, channel12Data, channel13Data = [newFeatureList[0]], [newFeatureList[1]], [newFeatureList[2]], [newFeatureList[3]], [newFeatureList[4]], [newFeatureList[5]], [newFeatureList[6]], [newFeatureList[7]], [newFeatureList[8]], [newFeatureList[9]], [newFeatureList[10]], [newFeatureList[11]], [newFeatureList[12]], [newFeatureList[13]]
        # print(channel0Data)
        channel0ArousalPredict, channel0ValencePredict = channel0ArousalSVM.predict(channel0Data)[0], channel0ValenceSVM.predict(channel0Data)[0]
        channel1ArousalPredict, channel1ValencePredict = channel1ArousalSVM.predict(channel1Data)[0], channel1ValenceSVM.predict(channel1Data)[0]
        channel2ArousalPredict, channel2ValencePredict = channel2ArousalSVM.predict(channel2Data)[0], channel2ValenceSVM.predict(channel2Data)[0]
        channel3ArousalPredict, channel3ValencePredict = channel3ArousalSVM.predict(channel3Data)[0], channel3ValenceSVM.predict(channel3Data)[0]
        channel4ArousalPredict, channel4ValencePredict = channel4ArousalSVM.predict(channel4Data)[0], channel4ValenceSVM.predict(channel4Data)[0]
        channel5ArousalPredict, channel5ValencePredict = channel5ArousalSVM.predict(channel5Data)[0], channel5ValenceSVM.predict(channel5Data)[0]
        channel6ArousalPredict, channel6ValencePredict = channel6ArousalSVM.predict(channel6Data)[0], channel6ValenceSVM.predict(channel6Data)[0]
        channel7ArousalPredict, channel7ValencePredict = channel7ArousalSVM.predict(channel7Data)[0], channel7ValenceSVM.predict(channel7Data)[0]
        channel8ArousalPredict, channel8ValencePredict = channel8ArousalSVM.predict(channel8Data)[0], channel8ValenceSVM.predict(channel8Data)[0]
        channel9ArousalPredict, channel9ValencePredict = channel9ArousalSVM.predict(channel9Data)[0], channel9ValenceSVM.predict(channel9Data)[0]
        channel10ArousalPredict, channel10ValencePredict = channel10ArousalSVM.predict(channel10Data)[0], channel10ValenceSVM.predict(channel10Data)[0]
        channel11ArousalPredict, channel11ValencePredict = channel11ArousalSVM.predict(channel11Data)[0], channel11ValenceSVM.predict(channel11Data)[0]
        channel12ArousalPredict, channel12ValencePredict = channel12ArousalSVM.predict(channel12Data)[0], channel12ValenceSVM.predict(channel12Data)[0]
        channel13ArousalPredict, channel13ValencePredict = channel13ArousalSVM.predict(channel13Data)[0], channel13ValenceSVM.predict(channel13Data)[0]
        # print(channel0ArousalPredict)
        # print(channel13ValencePredict)
        arousalList = [channel0ArousalPredict, channel1ArousalPredict, channel2ArousalPredict, channel3ArousalPredict, channel4ArousalPredict, channel5ArousalPredict, channel6ArousalPredict, channel7ArousalPredict,channel8ArousalPredict,channel9ArousalPredict,channel10ArousalPredict,channel11ArousalPredict,channel12ArousalPredict,channel13ArousalPredict]
        valenceList = [channel0ValencePredict, channel1ValencePredict, channel2ValencePredict,channel3ValencePredict,channel4ValencePredict,channel5ValencePredict,channel6ValencePredict,channel7ValencePredict,channel8ValencePredict,channel9ValencePredict,channel10ValencePredict,channel11ValencePredict,channel12ValencePredict,channel13ValencePredict]
        #8) get final emotion response
        if arousalList.count(1) >= arousalList.count(-1):
            arousal = 1
        else:
            arousal = -1
        if valenceList.count(1) >= valenceList.count(-1):
            valence = 1
        else:
            valence = -1
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
