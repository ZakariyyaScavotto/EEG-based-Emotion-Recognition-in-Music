from deepBeliefNetworkMaster.dbn import SupervisedDBNRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import statistics

#Notes on the format of the DEAP data
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
   
   #read in DE into a dictionary subjectPowers[subject#] = [list of DE]
    subjectPowers, tempSubjectPowersDict = {}, {}
    fullDEList = []
    for subjectNumber in range(1,33):
        subjectPowerFile = 'DEAPDiffEntropyALLBANDS\DEAPDiffEntropyALLBANDSSubject'+str(subjectNumber)+'.pickle'
        with open(subjectPowerFile, 'rb') as f:
            tempSubjectPowers = pickle.load(f)
        
        #v1 scaling: mean and std. dev per trial (both mean squared errors around 4 (3.9-4.4))
        fullSubjectDE = []
        for trial in tempSubjectPowers:
            meanDE, stdDevDE = statistics.mean(trial), statistics.pstdev(trial)
            fullSubjectDE.append([(de - meanDE)/stdDevDE + 0.5 for de in trial])
        subjectPowers[subjectNumber] = fullSubjectDE
        
        #v2 scaling: mean and std. dev across all trials per subject (both MSE around 4 (3.9-4.4) or NAN/inf. depending on run)
        # fullSubjectDE, originalTrialsList, everyDEList = [],[], []
        # for trial in tempSubjectPowers:
        #     originalTrialsList.append(trial)
        #     for de in trial:
        #         everyDEList.append(de)
        # meanSubject, stdDevSubject = statistics.mean(everyDEList), statistics.pstdev(everyDEList)
        # for trial in tempSubjectPowers:
        #     fullSubjectDE.append([(de - meanSubject)/stdDevSubject + 0.5 for de in trial])
        # subjectPowers[subjectNumber] = fullSubjectDE

    #v3 scaling: mean and std. dev across all subjects (DOESN'T WORK AT ALL)
    #     for trial in tempSubjectPowers:
    #         for de in trial:
    #             fullDEList.append(de)
    #     tempSubjectPowersDict[subjectNumber] = tempSubjectPowers
    # allSubjectMean, allSubjectStdDev = statistics.mean(fullDEList), statistics.pstdev(fullDEList)
    # for subjectNumber in range(1,33): 
    #     fullSubjectDE = []
    #     subjectPowersAllTrial = tempSubjectPowersDict[subjectNumber]
    #     for trial in subjectPowersAllTrial:
    #         fullSubjectDE.append([(de - allSubjectMean)/allSubjectStdDev + 0.5 for de in trial])
    #     subjectPowers[subjectNumber] = fullSubjectDE
    
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
            trialPower, trialLabel = np.array(subjectPow[trialNumber]), subjectLabel[trialNumber]
            #add powers to corresponding X lists
            arousalX.append(trialPower)
            valenceX.append(trialPower)
            # we now have the ability for continuity with a DBN, no need to classify as +- 1
            valenceY.append(trialLabel[0])
            arousalY.append(trialLabel[1])
    
   #split data into test/train sets for each DBN
    arousalXTrain, arousalXTest, arousalYTrain, arousalYTest = train_test_split(arousalX, arousalY, test_size = 0.3, random_state=42)
    valenceXTrain, valenceXTest, valenceYTrain, valenceYTest = train_test_split(valenceX, valenceY, test_size = 0.3, random_state=42)
    #converts to np arrays
    arousalXTrain, arousalXTest, arousalYTrain, arousalYTest, valenceXTrain, valenceXTest, valenceYTrain, valenceYTest = np.array(arousalXTrain), np.array(arousalXTest), np.array(arousalYTrain), np.array(arousalYTest), np.array(valenceXTrain), np.array(valenceXTest), np.array(valenceYTrain), np.array(valenceYTest)
   #creates/trains classifiers (since this is a DBN can now perform a regression task on continuous data)
    arousalClassifier = SupervisedDBNRegression(hidden_layers_structure = [32, 100,30,1],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=20, activation_function='relu')
    valenceClassifier = SupervisedDBNRegression(hidden_layers_structure = [32, 100,30,1],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=20, activation_function='relu')
    print('Starting Training')
    arousalClassifier.fit(arousalXTrain, arousalYTrain)
    print('Finished Arousal Training')
    valenceClassifier.fit(valenceXTrain, valenceYTrain)
    print('Finished Valence Training')

   #check accuracy, using mean squared error now that it's a regression
    arousalPredicts = arousalClassifier.predict(arousalXTest)
    valencePredicts = valenceClassifier.predict(valenceXTest)
    # print('Arousal Accuracy: ',metrics.accuracy_score(arousalYTest, arousalPredicts))
    print('Arousal Mean Squared Error: ',metrics.mean_squared_error(arousalYTest, arousalPredicts))
    # print('Valence Accuracy: ',metrics.accuracy_score(valenceYTest, valencePredicts))
    print('Valence Mean Squared Error: ',metrics.mean_squared_error(valenceYTest, valencePredicts))
    
   #end
    print('Done')

if __name__ == '__main__':
    main()