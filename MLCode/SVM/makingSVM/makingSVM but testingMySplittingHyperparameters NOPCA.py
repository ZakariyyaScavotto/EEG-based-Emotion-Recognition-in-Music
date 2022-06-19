import pickle #read in data
from sklearn.model_selection import train_test_split #used to split test data
from sklearn import svm #used to make svm
from sklearn import metrics #used to assess
from sklearn import preprocessing #used for preprocessing, scaling data
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
    
   #Scale X data to have a mean of 0 and std. deviation of 1
    arousalX, valenceX = [preprocessing.scale(channel) for channel in arousalX] , [preprocessing.scale(channel) for channel in valenceX]
    
   #split trials up to make it 2d
    arousalX2d, valenceX2d, arousalY2d, valenceY2d = [],[],[],[]
    for trial in range(len(arousalX)):
        for channel in arousalX[trial]:
            arousalX2d.append(channel)
            arousalY2d.append(arousalY[trial])
    for trial in range(len(valenceX)):
        for channel in valenceX[trial]:
            valenceX2d.append(channel)
            valenceY2d.append(valenceY[trial])

   #split data into test/train sets for each SVM
    arousalXTrain, arousalXTest, arousalYTrain, arousalYTest = train_test_split(arousalX2d, arousalY2d, test_size = 0.3, random_state=42)
    valenceXTrain, valenceXTest, valenceYTrain, valenceYTest = train_test_split(valenceX2d, valenceY2d, test_size = 0.3, random_state=42)
    
   #train each SVM
    arousalCLF = svm.SVC(kernel='rbf', C=1, gamma=0.01)
    valenceCLF = svm.SVC(kernel='rbf', C=0.1, gamma=1)
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
    
   #End
    print('Done')

if __name__=='__main__':
    main()