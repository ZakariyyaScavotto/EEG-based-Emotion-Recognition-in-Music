import pickle #read in data
from sklearn.model_selection import train_test_split #used to split test data
from sklearn import svm #used to make svm
from sklearn import metrics #used to assess
from sklearn import preprocessing #used for preprocessing, scaling data
from sklearn.decomposition import PCA
import numpy as np

#Notes on the format of the DEAP data
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
    
    #read in processed (Welched) data into a dictionary subjectPowers[subject#] = [list of powers for all trials]
    subjectPowers = {}
    for subjectNumber in range(1,33):
        subjectPowerFile = 'DEAPwelched\DEAPWelchSubject'+str(subjectNumber)+'.pickle'
        with open(subjectPowerFile, 'rb') as f:
            tempSubjectPowers = pickle.load(f)
        subjectPowers[subjectNumber] = tempSubjectPowers
    
    #read in labels
    subjectLabels = {}
    for subjectNumber in range(1,33):
        if subjectNumber < 10:
            fileName = 'preprocessing code\DEAP_data_preprocessed_python\s0'+str(subjectNumber)+'.dat'
        else:
            fileName = 'preprocessing code\DEAP_data_preprocessed_python\s'+str(subjectNumber)+'.dat'
        with open(fileName,'rb') as f:
            dataSubject = pickle.load(f, encoding='latin1') 
        subjectLabels[subjectNumber] = dataSubject['labels']
    
    #duplicate to train on valence, arousal (for each make separate training labels (just arrays of the singular valence/arousal #))
    arousalX, arousalY, valenceX, valenceY = [], [], [], []
    for subjectNumber in range(1,33):
        subjectPow, subjectLabel = subjectPowers[subjectNumber],subjectLabels[subjectNumber] 
        for trialNumber in range(40):
            trialPower, trialLabel = subjectPow[trialNumber], subjectLabel[trialNumber]
            #add powers to corresponding X lists
            arousalX.append(trialPower)
            valenceX.append(trialPower)
            #add labels (+- 1 ) to corresponding Y lists depending if given label is > or < 5 (midpoint of rating)
            if trialLabel[0] > 5:
                valenceY.append(1)
            else:
                valenceY.append(-1)
            if trialLabel[1] > 5:
                arousalY.append(1)
            else:
                arousalY.append(-1)
    
    #Scale X data, should i be doing this??
    arousalX, valenceX = [preprocessing.scale(channel) for channel in arousalX] , [preprocessing.scale(channel) for channel in valenceX]
    
    #converts to numpy array
    arousalXArray, arousalYArray, valenceXArray, valenceYArray = np.array(arousalX), np.array(arousalY), np.array(valenceX), np.array(valenceY)
    del arousalX, arousalY, valenceX, valenceY #clears the original lists from memory since they are relatively large
    
    #PCA
    arousalX2d, valenceX2d = np.array([x.flatten() for x in arousalXArray]), np.array([x.flatten() for x in valenceXArray])
    #^^^^ needed to reduce dimensionality for PCA, but what is this doing to the data?
    pcaArousalX, pcaValenceX = PCA(n_components=3), PCA(n_components=3)
    pcaArousalX.fit(arousalX2d)
    pcaValenceX.fit(valenceX2d)
    pcaArousalXFinal, pcaValenceXFinal = pcaArousalX.transform(arousalX2d), pcaValenceX.transform(valenceX2d)

    #split data into test/train sets for each SVM
    arousalXTrain, arousalXTest, arousalYTrain, arousalYTest = train_test_split(pcaArousalXFinal, arousalYArray, test_size = 0.3, random_state=42)
    valenceXTrain, valenceXTest, valenceYTrain, valenceYTest = train_test_split(pcaValenceXFinal, valenceYArray, test_size = 0.3, random_state=42)
    
    #train each SVM
    arousalCLF = svm.SVC(kernel='rbf')
    valenceCLF = svm.SVC(kernel='rbf')
    print('Starting Training')
    arousalCLF.fit(arousalXTrain, arousalYTrain)
    print('Finished training arousal')
    valenceCLF.fit(valenceXTrain, valenceYTrain)
    print('Finished training valence')
    
    #look at accuracies
    arousalPredicts = arousalCLF.predict(arousalXTest)
    valencePredicts = valenceCLF.predict(valenceXTest)
    print('Arousal Accuracy: ',metrics.accuracy_score(arousalYTest, arousalPredicts))
    print('Valence Accuracy: ',metrics.accuracy_score(valenceYTest, valencePredicts))
    
    print('Done')

if __name__=='__main__':
    main()