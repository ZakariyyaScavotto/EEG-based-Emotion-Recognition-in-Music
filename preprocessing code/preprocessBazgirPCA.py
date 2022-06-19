import pickle
from statistics import mean
import numpy as np
import pywt
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # Read in EEG data
    # subjectNumber = int(input('Type subject number 1-32: '))
    for subjectNumber in range(1,33):
        subjectDataDict = {}
        if subjectNumber < 10:
            fileName = 'preprocessing code\DEAP_data_preprocessed_python\s0'+str(subjectNumber)+'.dat'
        else:
            fileName = 'preprocessing code\DEAP_data_preprocessed_python\s'+str(subjectNumber)+'.dat'
        with open(fileName,'rb') as f:
            dataSubject = pickle.load(f, encoding='latin1')
        rawSubjectDataList, allTrialList = dataSubject['data'], []
        for trial in rawSubjectDataList:
            allTrialList.append(trial[:32])
        subjectDataDict[subjectNumber] = allTrialList

        # Average mean reference and normalization
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
            # 8064 points, 60 second trial -> 2 seconds = 268.8 (268) points
            # 60 windows per channel
        windowedSignalDict = {}
        subjectWindowedData = []
        twoSeconds = 268
        fourSeconds = twoSeconds*2
        print('Starting windowing')
        for trial in subjectDataDict[subjectNumber]:
            trialList = []
            #2 second windows to start
            for channel in trial:
                startIndex, endIndex = 0, twoSeconds
                channelList = []
                while endIndex < 8064:
                    channelList.append(channel[startIndex:endIndex])
                    startIndex += twoSeconds//2
                    endIndex += twoSeconds//2 # want 50% overlap
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
        print('Starting scaling')
        scaledSubjectFeatures = []
        for trial in subjectFeatureList:
            trialList = []
            for channel in trial:
                transformedChannel = StandardScaler().fit_transform(channel)
                trialList.append(transformedChannel)
            scaledSubjectFeatures.append(trialList)
        # PCA
        print('Starting PCA')
        pcaSubjectFeatures = []
        pca = PCA()
        for trial in scaledSubjectFeatures:
            trialList = []
            for channel in trial:
                pcaChannel = pca.fit_transform(channel)
                trialList.append(pcaChannel)
            pcaSubjectFeatures.append(trialList)
        # save
        pickledFileName = 'DEAPBazgirPCA'+str(subjectNumber)+'.pickle'
        with open(pickledFileName, 'wb') as f:
            pickle.dump(subjectFeatureList, f)

        print('Done with subject ',subjectNumber)

    print('Done')

if __name__ == '__main__':
    main()