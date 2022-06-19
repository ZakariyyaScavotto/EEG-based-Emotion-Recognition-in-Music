import pickle
import scipy.signal as signal
# import numpy as np

#Notes on the format of the data folder
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
    for subjectNumber in range(1,33):
        #testing with participant 1, reading in participant 1's data
        if subjectNumber < 10:
            fileName = 'preprocessing code\DEAP_data_preprocessed_python\s0'+str(subjectNumber)+'.dat'
        else:
            fileName = 'preprocessing code\DEAP_data_preprocessed_python\s'+str(subjectNumber)+'.dat'
        with open(fileName,'rb') as f:
            dataSubject = pickle.load(f, encoding='latin1')
        EEGlists = []
        for trialNum in range(40):
            trialList = []
            for i in range(32):
                frequencies, powers = signal.welch(dataSubject['data'][trialNum][i], fs=128, nperseg=1024)
                # print(np.where(frequencies==50))
                trialList.append(powers)
            EEGlists.append(trialList)
        pickledFileName = 'DEAPWelchSubject'+str(subjectNumber)+'.pickle'
        with open(pickledFileName, 'wb') as f:
            pickle.dump(EEGlists, f)
    
    print("Done")

if __name__ == "__main__":
    main()