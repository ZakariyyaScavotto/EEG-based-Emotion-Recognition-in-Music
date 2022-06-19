import pickle
from scipy.stats import differential_entropy

#Notes on the format of the data folder
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
    for subjectNumber in range(1,33):
        fileName = 'DEAPwelched\DEAPWelchSubject'+str(subjectNumber)+'.pickle'
        with open(fileName,'rb') as f:
            dataSubject = pickle.load(f, encoding='latin1')
        DEList = []
        for trialNum in range(40):
            trialList = []
            for i in range(32):
                #Gets differential entropy of all frequency bands
                tempDE = differential_entropy(dataSubject[trialNum][i])
                trialList.append(tempDE)
            DEList.append(trialList)
        pickledFileName = 'DEAPDiffEntropyALLBANDSSubject'+str(subjectNumber)+'.pickle'
        with open(pickledFileName, 'wb') as f:
            pickle.dump(DEList, f)
    
    print("Done")

if __name__ == "__main__":
    main()