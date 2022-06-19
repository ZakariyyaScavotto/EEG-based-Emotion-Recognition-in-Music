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
                channelData = dataSubject[trialNum][i]
                # separate each frequency into its own list
                # delta 1-4, theta 4-7, alpha 7-13, beta 13-30, gamma 30-50
                # in powers list goes up by .125Hz every time
                delta, theta, alpha = channelData[8:31],channelData[32:55],channelData[56:103],
                beta, gamma = channelData[104:239],channelData[240:400]
                # get DE of that frequency, add to list
                DEdelta, DEtheta, DEalpha = differential_entropy(delta), differential_entropy(theta), differential_entropy(alpha)
                DEbeta, DEgamma = differential_entropy(beta), differential_entropy(gamma)
                trialList.append(DEdelta)
                trialList.append(DEtheta)
                trialList.append(DEalpha)
                trialList.append(DEbeta)
                trialList.append(DEgamma)
            DEList.append(trialList)
        pickledFileName = 'DEAPDiffEntropySUBBANDSSubject'+str(subjectNumber)+'.pickle'
        with open(pickledFileName, 'wb') as f:
            pickle.dump(DEList, f)
    
    print("Done")

if __name__ == "__main__":
    main()