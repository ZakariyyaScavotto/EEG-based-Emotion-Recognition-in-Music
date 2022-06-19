import pickle #read in data
from sklearn.model_selection import train_test_split #used to split test data
from sklearn import svm #used to make svm
from sklearn import metrics #used to assess
from sklearn import preprocessing #used for preprocessing, scaling data
import numpy as np
from sklearn.model_selection import GridSearchCV #used to get hyperparameters

#Notes on the format of the DEAP data
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
    
   #read in DE into a dictionary subjectPowers[subject#] = [list of powers for all trials]
    subjectPowers = {}
    for subjectNumber in range(1,33):
        subjectPowerFile = 'DEAPDiffEntropyALLBANDS\DEAPDiffEntropyALLBANDSSubject'+str(subjectNumber)+'.pickle'
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
    
   #split data into test/train sets for each SVM
    arousalXTrain, arousalXTest, arousalYTrain, arousalYTest = train_test_split(arousalX, arousalY, test_size = 0.3, random_state=42)
    valenceXTrain, valenceXTest, valenceYTrain, valenceYTest = train_test_split(valenceX, valenceY, test_size = 0.3, random_state=42)
    
   #gridsearch
    paramGridArousal = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}
    gridArousal = GridSearchCV(svm.SVC(), paramGridArousal, refit=True, verbose=2)
    gridArousal.fit(arousalXTrain, arousalYTrain)
    print('grid arousal results')
    print(gridArousal.best_estimator_)

    paramGridValence = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}
    gridValence = GridSearchCV(svm.SVC(), paramGridValence, refit=True, verbose=2)
    gridValence.fit(valenceXTrain, valenceYTrain)
    print('grid valence results')
    print(gridValence.best_estimator_)

   #End
    print('Done')

if __name__=='__main__':
    main()