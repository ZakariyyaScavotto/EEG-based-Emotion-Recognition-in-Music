import pickle
from statistics import mean
import numpy as np
import pywt
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mne.io import read_raw_bdf

def main():
    # Read in EEG data
    subjectDataDict = {}
    for subjectNumber in range(1,24):
        if subjectNumber < 10:
            filename = 'preprocessing code\DEAP_data_original\s0'+str(subjectNumber)+'.bdf'
        else:
            filename = 'preprocessing code\DEAP_data_original\s'+str(subjectNumber)+'.bdf'
        rawBDF = read_raw_bdf(filename)
        dataFrame = rawBDF.to_data_frame()
        # testing for subjects 24-32
        statusData = rawBDF.get_data(picks=47)[0]
        statusStartOfPlayback = np.where(np.array(statusData) == 4)[0]
        statusFixationAfterPlayback = np.where(np.array(statusData) == 5)[0]
        # end testing
        AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 = dataFrame.AF3.values, dataFrame.F7.values, dataFrame.F3.values, dataFrame.FC5.values, dataFrame.T7.values, dataFrame.P7.values, dataFrame.O1.values, dataFrame.O2.values, dataFrame.P8.values, dataFrame.T8.values, dataFrame.FC6.values, dataFrame.F4.values, dataFrame.F8.values, dataFrame.AF4.values
        status = dataFrame.Status.values
        statusStartOfPlayback = np.where(np.array(status) == 4)[0]
        statusFixationAfterPlayback = np.where(np.array(status) == 5)[0]

        # Separate into trials
        subjectTrials = []
        for trial in range(39):
            trialList = []
            marker = 6*trial
            changed = False
            while marker > len(statusStartOfPlayback-1):
                marker = marker-1
                changed = True
            while marker > len(statusFixationAfterPlayback-1):
                marker = marker-1
                changed = True
            if (changed):
                marker = marker-1
            startIndex, endIndex = statusStartOfPlayback[marker], statusFixationAfterPlayback[marker]
            if startIndex < endIndex:
                trialList.append(AF3[startIndex:endIndex])
                trialList.append(F7[startIndex:endIndex])
                trialList.append(F3[startIndex:endIndex])
                trialList.append(FC5[startIndex:endIndex])
                trialList.append(T7[startIndex:endIndex])
                trialList.append(P7[startIndex:endIndex])
                trialList.append(O1[startIndex:endIndex])
                trialList.append(O2[startIndex:endIndex])
                trialList.append(P8[startIndex:endIndex])
                trialList.append(T8[startIndex:endIndex])
                trialList.append(FC6[startIndex:endIndex])
                trialList.append(F4[startIndex:endIndex])
                trialList.append(F8[startIndex:endIndex])
                trialList.append(AF4[startIndex:endIndex])
                subjectTrials.append(trialList)
        subjectDataDict[subjectNumber] = subjectTrials
        # do the processing

        # Average mean reference and normalization
        print('Starting mean referencing')
        newDataSubject = []
        for trial in subjectDataDict[subjectNumber]:
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
        subjectDataDict[subjectNumber] = newDataSubject

        # Apply Discrete Wavelet Transform (DWT)
        # 1) window EEG signals
            # 30750 points, 60 second trial -> 2 seconds = 512.5 (512) points
        windowedSignalDict = {}
        subjectWindowedData = []
        twoSeconds = 512
        fourSeconds = twoSeconds*2
        print('Starting windowing')
        for trial in subjectDataDict[subjectNumber]:
            trialList = []
            #2 second windows to start
            for channel in trial:
                startIndex, endIndex = 0, twoSeconds
                channelList = []
                while endIndex < 30750:
                    if len(channel[startIndex:endIndex]) != 0:
                        channelList.append(channel[startIndex:endIndex])
                    startIndex += twoSeconds//2
                    endIndex += twoSeconds//2 # want 50% overlap
                if len(channel[startIndex:]) != 0:
                    channelList.append(channel[startIndex:])
                trialList.append(channelList)
            subjectWindowedData.append(trialList)
        windowedSignalDict[subjectNumber] = subjectWindowedData
        # 2) decompose signals into frequency bands by db4 mother wavelet function
        print('Starting DWT computations')
        subjectDWT = []
        for trial in windowedSignalDict[subjectNumber]:
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
        # Get Entropy (and energy?) of every frequency band from every window
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
        # First, scale data before PCA
        
        # PCA
        
        # save
        # pickledFileName = 'RAWDEAPBazgir'+str(subjectNumber)+'.pickle'
        # with open(pickledFileName, 'wb') as f:
        #     pickle.dump(subjectFeatureList, f)

        print('Done with subject ',subjectNumber)
    print('Done')

if __name__ == '__main__':
    main()