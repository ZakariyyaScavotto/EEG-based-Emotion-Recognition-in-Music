from deepBeliefNetworkMaster.dbn import SupervisedDBNRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import statistics

#Notes on the format of the DEAP data
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
    #read in DE into a dictionary subjectPowers[subject#] = [list of powers for all trials]
    subjectPowers = {}
    for subjectNumber in range(1,33):
        subjectPowerFile = 'DEAPBazgirPCA\DEAPBazgirPCA'+str(subjectNumber)+'.pickle'
        with open(subjectPowerFile, 'rb') as f:
            tempSubjectPowers = pickle.load(f)
        subjectPowers[subjectNumber] = tempSubjectPowers

    #cut down to the 14 channels
    # 32 subject > each has 40 trials > each  has 14 channels > each has 60 windows > each has feature list
    # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    emotivChannels = [1, 3, 2, 4, 7, 11, 13, 31, 29, 26, 22, 19, 20, 17]
    subjectPowersFinal = {}
    for subjectNumber in range(1,33):
        subjectList = []
        for trial in range(40):
            trialChannels = subjectPowers[subjectNumber][trial]
            finalizedTrial = []
            for channelInd in emotivChannels:
                finalizedTrial.append(trialChannels[channelInd])
            subjectList.append(finalizedTrial)
        subjectPowersFinal[subjectNumber] = subjectList

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

    #separate into channels? pull channel out from each trial for each subject, have corresponding label dict to go off of?
    channelPowers = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    channelLabels = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    for subjectNumber in range(1,33):
        subjectList = subjectPowersFinal[subjectNumber]
        for trialInd in range(40):
            trialList = subjectList[trialInd]
            for channelInd in range(14):
                trialFeatures = trialList[channelInd]
                unzipped = []
                for window in trialFeatures:
                    for element in window:
                        unzipped.append(element)
                channelPowers[channelInd].append(unzipped)
                channelLabels[channelInd].append(subjectLabels[subjectNumber][trialInd])

    #duplicate to train on valence, arousal (for each make separate training labels (just arrays of the singular valence/arousal #))
    # THE Xs ARE THE SAME FOR VALENCE AND AROUSAL, SO NO NEED TO DUPLICATE
    # ONLY NEED TO ALTER THE Y-LISTS SO THAT THEY ARE BINARY
    arousalY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    valenceY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    for channelInd in range(14):
        channelLabel = channelLabels[channelInd]
        for trialNum in range(1280):
            trialLabel = channelLabel[trialNum]
            valenceY[channelInd].append(trialLabel[0])
            arousalY[channelInd].append(trialLabel[1])
    
   #split data into test/train sets for each DBN (and each channel), then convert to np arrays
    print('Done processing, splitting')
    channelPowersTrain1, channelPowersTest1, arousalYTrain1, arousalYTest1 = train_test_split(channelPowers[0], arousalY[0], test_size = 0.3, random_state=42)
    channelPowersTrain1, channelPowersTest1, valenceYTrain1, valenceYTest1 = train_test_split(channelPowers[0], valenceY[0], test_size = 0.3, random_state=42)
    channelPowersTrain1, channelPowersTest1, arousalYTrain1, arousalYTest1, channelPowersTrain1, channelPowersTest1, valenceYTrain1, valenceYTest1 = np.array(channelPowersTrain1), np.array(channelPowersTest1), np.array(arousalYTrain1), np.array(arousalYTest1), np.array(channelPowersTrain1), np.array(channelPowersTest1), np.array(valenceYTrain1), np.array(valenceYTest1)

    channelPowersTrain2, channelPowersTest2, arousalYTrain2, arousalYTest2 = train_test_split(channelPowers[1], arousalY[1], test_size = 0.3, random_state=42)
    channelPowersTrain2, channelPowersTest2, valenceYTrain2, valenceYTest2 = train_test_split(channelPowers[1], valenceY[1], test_size = 0.3, random_state=42)
    channelPowersTrain2, channelPowersTest2, arousalYTrain2, arousalYTest2, channelPowersTrain2, channelPowersTest2, valenceYTrain2, valenceYTest2 = np.array(channelPowersTrain2), np.array(channelPowersTest2), np.array(arousalYTrain2), np.array(arousalYTest2), np.array(channelPowersTrain2), np.array(channelPowersTest2), np.array(valenceYTrain2), np.array(valenceYTest2)

    channelPowersTrain3, channelPowersTest3, arousalYTrain3, arousalYTest3 = train_test_split(channelPowers[2], arousalY[2], test_size = 0.3, random_state=42)
    channelPowersTrain3, channelPowersTest3, valenceYTrain3, valenceYTest3 = train_test_split(channelPowers[2], valenceY[2], test_size = 0.3, random_state=42)
    channelPowersTrain3, channelPowersTest3, arousalYTrain3, arousalYTest3, channelPowersTrain3, channelPowersTest3, valenceYTrain3, valenceYTest3 = np.array(channelPowersTrain3), np.array(channelPowersTest3), np.array(arousalYTrain3), np.array(arousalYTest3), np.array(channelPowersTrain3), np.array(channelPowersTest3), np.array(valenceYTrain3), np.array(valenceYTest3)

    channelPowersTrain4, channelPowersTest4, arousalYTrain4, arousalYTest4 = train_test_split(channelPowers[3], arousalY[3], test_size = 0.3, random_state=42)
    channelPowersTrain4, channelPowersTest4, valenceYTrain4, valenceYTest4 = train_test_split(channelPowers[3], valenceY[3], test_size = 0.3, random_state=42)
    channelPowersTrain4, channelPowersTest4, arousalYTrain4, arousalYTest4, channelPowersTrain4, channelPowersTest4, valenceYTrain4, valenceYTest4 = np.array(channelPowersTrain4), np.array(channelPowersTest4), np.array(arousalYTrain4), np.array(arousalYTest4), np.array(channelPowersTrain4), np.array(channelPowersTest4), np.array(valenceYTrain4), np.array(valenceYTest4)

    channelPowersTrain5, channelPowersTest5, arousalYTrain5, arousalYTest5 = train_test_split(channelPowers[4], arousalY[4], test_size = 0.3, random_state=42)
    channelPowersTrain5, channelPowersTest5, valenceYTrain5, valenceYTest5 = train_test_split(channelPowers[4], valenceY[4], test_size = 0.3, random_state=42)
    channelPowersTrain5, channelPowersTest5, arousalYTrain5, arousalYTest5, channelPowersTrain5, channelPowersTest5, valenceYTrain5, valenceYTest5 = np.array(channelPowersTrain5), np.array(channelPowersTest5), np.array(arousalYTrain5), np.array(arousalYTest5), np.array(channelPowersTrain5), np.array(channelPowersTest5), np.array(valenceYTrain5), np.array(valenceYTest5)

    channelPowersTrain6, channelPowersTest6, arousalYTrain6, arousalYTest6 = train_test_split(channelPowers[5], arousalY[5], test_size = 0.3, random_state=42)
    channelPowersTrain6, channelPowersTest6, valenceYTrain6, valenceYTest6 = train_test_split(channelPowers[5], valenceY[5], test_size = 0.3, random_state=42)
    channelPowersTrain6, channelPowersTest6, arousalYTrain6, arousalYTest6, channelPowersTrain6, channelPowersTest6, valenceYTrain6, valenceYTest6 = np.array(channelPowersTrain6), np.array(channelPowersTest6), np.array(arousalYTrain6), np.array(arousalYTest6), np.array(channelPowersTrain6), np.array(channelPowersTest6), np.array(valenceYTrain6), np.array(valenceYTest6)

    channelPowersTrain7, channelPowersTest7, arousalYTrain7, arousalYTest7 = train_test_split(channelPowers[6], arousalY[6], test_size = 0.3, random_state=42)
    channelPowersTrain7, channelPowersTest7, valenceYTrain7, valenceYTest7 = train_test_split(channelPowers[6], valenceY[6], test_size = 0.3, random_state=42)
    channelPowersTrain7, channelPowersTest7, arousalYTrain7, arousalYTest7, channelPowersTrain7, channelPowersTest7, valenceYTrain7, valenceYTest7 = np.array(channelPowersTrain7), np.array(channelPowersTest7), np.array(arousalYTrain7), np.array(arousalYTest7), np.array(channelPowersTrain7), np.array(channelPowersTest7), np.array(valenceYTrain7), np.array(valenceYTest7)

    channelPowersTrain8, channelPowersTest8, arousalYTrain8, arousalYTest8 = train_test_split(channelPowers[7], arousalY[7], test_size = 0.3, random_state=42)
    channelPowersTrain8, channelPowersTest8, valenceYTrain8, valenceYTest8 = train_test_split(channelPowers[7], valenceY[7], test_size = 0.3, random_state=42)
    channelPowersTrain8, channelPowersTest8, arousalYTrain8, arousalYTest8, channelPowersTrain8, channelPowersTest8, valenceYTrain8, valenceYTest8 = np.array(channelPowersTrain8), np.array(channelPowersTest8), np.array(arousalYTrain8), np.array(arousalYTest8), np.array(channelPowersTrain8), np.array(channelPowersTest8), np.array(valenceYTrain8), np.array(valenceYTest8)

    channelPowersTrain9, channelPowersTest9, arousalYTrain9, arousalYTest9 = train_test_split(channelPowers[8], arousalY[8], test_size = 0.3, random_state=42)
    channelPowersTrain9, channelPowersTest9, valenceYTrain9, valenceYTest9 = train_test_split(channelPowers[8], valenceY[8], test_size = 0.3, random_state=42)
    channelPowersTrain9, channelPowersTest9, arousalYTrain9, arousalYTest9, channelPowersTrain9, channelPowersTest9, valenceYTrain9, valenceYTest9 = np.array(channelPowersTrain9), np.array(channelPowersTest9), np.array(arousalYTrain9), np.array(arousalYTest9), np.array(channelPowersTrain9), np.array(channelPowersTest9), np.array(valenceYTrain9), np.array(valenceYTest9)

    channelPowersTrain10, channelPowersTest10, arousalYTrain10, arousalYTest10 = train_test_split(channelPowers[9], arousalY[9], test_size = 0.3, random_state=42)
    channelPowersTrain10, channelPowersTest10, valenceYTrain10, valenceYTest10 = train_test_split(channelPowers[9], valenceY[9], test_size = 0.3, random_state=42)
    channelPowersTrain10, channelPowersTest10, arousalYTrain10, arousalYTest10, channelPowersTrain10, channelPowersTest10, valenceYTrain10, valenceYTest10 = np.array(channelPowersTrain10), np.array(channelPowersTest10), np.array(arousalYTrain10), np.array(arousalYTest10), np.array(channelPowersTrain10), np.array(channelPowersTest10), np.array(valenceYTrain10), np.array(valenceYTest10)

    channelPowersTrain11, channelPowersTest11, arousalYTrain11, arousalYTest11 = train_test_split(channelPowers[10], arousalY[10], test_size = 0.3, random_state=42)
    channelPowersTrain11, channelPowersTest11, valenceYTrain11, valenceYTest11 = train_test_split(channelPowers[10], valenceY[10], test_size = 0.3, random_state=42)
    channelPowersTrain11, channelPowersTest11, arousalYTrain11, arousalYTest11, channelPowersTrain11, channelPowersTest11, valenceYTrain11, valenceYTest11 = np.array(channelPowersTrain11), np.array(channelPowersTest11), np.array(arousalYTrain11), np.array(arousalYTest11), np.array(channelPowersTrain11), np.array(channelPowersTest11), np.array(valenceYTrain11), np.array(valenceYTest11)

    channelPowersTrain12, channelPowersTest12, arousalYTrain12, arousalYTest12 = train_test_split(channelPowers[11], arousalY[11], test_size = 0.3, random_state=42)
    channelPowersTrain12, channelPowersTest12, valenceYTrain12, valenceYTest12 = train_test_split(channelPowers[11], valenceY[11], test_size = 0.3, random_state=42)
    channelPowersTrain12, channelPowersTest12, arousalYTrain12, arousalYTest12, channelPowersTrain12, channelPowersTest12, valenceYTrain12, valenceYTest12 = np.array(channelPowersTrain12), np.array(channelPowersTest12), np.array(arousalYTrain12), np.array(arousalYTest12), np.array(channelPowersTrain12), np.array(channelPowersTest12), np.array(valenceYTrain12), np.array(valenceYTest12)

    channelPowersTrain13, channelPowersTest13, arousalYTrain13, arousalYTest13 = train_test_split(channelPowers[12], arousalY[12], test_size = 0.3, random_state=42)
    channelPowersTrain13, channelPowersTest13, valenceYTrain13, valenceYTest13 = train_test_split(channelPowers[12], valenceY[12], test_size = 0.3, random_state=42)
    channelPowersTrain13, channelPowersTest13, arousalYTrain13, arousalYTest13, channelPowersTrain13, channelPowersTest13, valenceYTrain13, valenceYTest13 = np.array(channelPowersTrain13), np.array(channelPowersTest13), np.array(arousalYTrain13), np.array(arousalYTest13), np.array(channelPowersTrain13), np.array(channelPowersTest13), np.array(valenceYTrain13), np.array(valenceYTest13)

    channelPowersTrain14, channelPowersTest14, arousalYTrain14, arousalYTest14 = train_test_split(channelPowers[13], arousalY[13], test_size = 0.3, random_state=42)
    channelPowersTrain14, channelPowersTest14, valenceYTrain14, valenceYTest14 = train_test_split(channelPowers[13], valenceY[13], test_size = 0.3, random_state=42)
    channelPowersTrain14, channelPowersTest14, arousalYTrain14, arousalYTest14, channelPowersTrain14, channelPowersTest14, valenceYTrain14, valenceYTest14 = np.array(channelPowersTrain14), np.array(channelPowersTest14), np.array(arousalYTrain14), np.array(arousalYTest14), np.array(channelPowersTrain14), np.array(channelPowersTest14), np.array(valenceYTrain14), np.array(valenceYTest14)

   #creates/trains classifiers (since this is a DBN can now perform a regression task on continuous data)
    arousalClassifier1 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier1 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier1.fit(channelPowersTrain1, arousalYTrain1)
    valenceClassifier1.fit(channelPowersTrain1, valenceYTrain1)
    arousalPredicts1 = arousalClassifier1.predict(channelPowersTest1)
    valencePredicts1 = valenceClassifier1.predict(channelPowersTest1)
    print('Arousal1 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest1, arousalPredicts1))+'\n')
    print('Arousal1 r^2: '+ str(metrics.r2_score(arousalYTest1, arousalPredicts1))+'\n')
    print('Valence1 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest1, valencePredicts1))+'\n')
    print('Valence1 r^2: '+ str(metrics.r2_score(valenceYTest1, valencePredicts1))+'\n')
    print('\n')

    arousalClassifier2 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier2 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier2.fit(channelPowersTrain2, arousalYTrain2)
    valenceClassifier2.fit(channelPowersTrain2, valenceYTrain2)
    arousalPredicts2 = arousalClassifier2.predict(channelPowersTest2)
    valencePredicts2 = valenceClassifier2.predict(channelPowersTest2)
    print('Arousal2 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest2, arousalPredicts2))+'\n')
    print('Arousal2 r^2: '+ str(metrics.r2_score(arousalYTest2, arousalPredicts2))+'\n')
    print('Valence2 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest2, valencePredicts2))+'\n')
    print('Valence2 r^2: '+ str(metrics.r2_score(valenceYTest2, valencePredicts2))+'\n')
    print('\n')

    arousalClassifier3 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier3 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier3.fit(channelPowersTrain3, arousalYTrain3)
    valenceClassifier3.fit(channelPowersTrain3, valenceYTrain3)
    arousalPredicts3 = arousalClassifier3.predict(channelPowersTest3)
    valencePredicts3 = valenceClassifier3.predict(channelPowersTest3)
    print('Arousal3 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest3, arousalPredicts3))+'\n')
    print('Arousal3 r^2: '+ str(metrics.r2_score(arousalYTest3, arousalPredicts3))+'\n')
    print('Valence3 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest3, valencePredicts3))+'\n')
    print('Valence3 r^2: '+ str(metrics.r2_score(valenceYTest3, valencePredicts3))+'\n')
    print('\n')

    arousalClassifier4 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier4 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier4.fit(channelPowersTrain4, arousalYTrain4)
    valenceClassifier4.fit(channelPowersTrain4, valenceYTrain4)
    arousalPredicts4 = arousalClassifier4.predict(channelPowersTest4)
    valencePredicts4 = valenceClassifier4.predict(channelPowersTest4)
    print('Arousal4 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest4, arousalPredicts4))+'\n')
    print('Arousal4 r^2: '+ str(metrics.r2_score(arousalYTest4, arousalPredicts4))+'\n')
    print('Valence4 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest4, valencePredicts4))+'\n')
    print('Valence4 r^2: '+ str(metrics.r2_score(valenceYTest4, valencePredicts4))+'\n')
    print('\n')

    arousalClassifier5 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier5 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier5.fit(channelPowersTrain5, arousalYTrain5)
    valenceClassifier5.fit(channelPowersTrain5, valenceYTrain5)
    arousalPredicts5 = arousalClassifier5.predict(channelPowersTest5)
    valencePredicts5 = valenceClassifier5.predict(channelPowersTest5)
    print('Arousal5 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest5, arousalPredicts5))+'\n')
    print('Arousal5 r^2: '+ str(metrics.r2_score(arousalYTest5, arousalPredicts5))+'\n')
    print('Valence5 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest5, valencePredicts5))+'\n')
    print('Valence5 r^2: '+ str(metrics.r2_score(valenceYTest5, valencePredicts5))+'\n')
    print('\n')

    arousalClassifier6 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier6 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier6.fit(channelPowersTrain6, arousalYTrain6)
    valenceClassifier6.fit(channelPowersTrain6, valenceYTrain6)
    arousalPredicts6 = arousalClassifier6.predict(channelPowersTest6)
    valencePredicts6 = valenceClassifier6.predict(channelPowersTest6)
    print('Arousal6 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest6, arousalPredicts6))+'\n')
    print('Arousal6 r^2: '+ str(metrics.r2_score(arousalYTest6, arousalPredicts6))+'\n')
    print('Valence6 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest6, valencePredicts6))+'\n')
    print('Valence6 r^2: '+ str(metrics.r2_score(valenceYTest6, valencePredicts6))+'\n')
    print('\n')

    arousalClassifier7 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier7 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier7.fit(channelPowersTrain7, arousalYTrain7)
    valenceClassifier7.fit(channelPowersTrain7, valenceYTrain7)
    arousalPredicts7 = arousalClassifier7.predict(channelPowersTest7)
    valencePredicts7 = valenceClassifier7.predict(channelPowersTest7)
    print('Arousal7 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest7, arousalPredicts7))+'\n')
    print('Arousal7 r^2: '+ str(metrics.r2_score(arousalYTest7, arousalPredicts7))+'\n')
    print('Valence7 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest7, valencePredicts7))+'\n')
    print('Valence7 r^2: '+ str(metrics.r2_score(valenceYTest7, valencePredicts7))+'\n')
    print('\n')

    arousalClassifier8 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier8 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier8.fit(channelPowersTrain8, arousalYTrain8)
    valenceClassifier8.fit(channelPowersTrain8, valenceYTrain8)
    arousalPredicts8 = arousalClassifier8.predict(channelPowersTest8)
    valencePredicts8 = valenceClassifier8.predict(channelPowersTest8)
    print('Arousal8 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest8, arousalPredicts8))+'\n')
    print('Arousal8 r^2: '+ str(metrics.r2_score(arousalYTest8, arousalPredicts8))+'\n')
    print('Valence8 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest8, valencePredicts8))+'\n')
    print('Valence8 r^2: '+ str(metrics.r2_score(valenceYTest8, valencePredicts8))+'\n')
    print('\n')

    arousalClassifier9 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier9 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier9.fit(channelPowersTrain9, arousalYTrain9)
    valenceClassifier9.fit(channelPowersTrain9, valenceYTrain9)
    arousalPredicts9 = arousalClassifier9.predict(channelPowersTest9)
    valencePredicts9 = valenceClassifier9.predict(channelPowersTest9)
    print('Arousal9 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest9, arousalPredicts9))+'\n')
    print('Arousal9 r^2: '+ str(metrics.r2_score(arousalYTest9, arousalPredicts9))+'\n')
    print('Valence9 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest9, valencePredicts9))+'\n')
    print('Valence9 r^2: '+ str(metrics.r2_score(valenceYTest9, valencePredicts9))+'\n')
    print('\n')

    arousalClassifier10 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier10 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier10.fit(channelPowersTrain10, arousalYTrain10)
    valenceClassifier10.fit(channelPowersTrain10, valenceYTrain10)
    arousalPredicts10 = arousalClassifier10.predict(channelPowersTest10)
    valencePredicts10 = valenceClassifier10.predict(channelPowersTest10)
    print('Arousal10 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest10, arousalPredicts10))+'\n')
    print('Arousal10 r^2: '+ str(metrics.r2_score(arousalYTest10, arousalPredicts10))+'\n')
    print('Valence10 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest10, valencePredicts10))+'\n')
    print('Valence10 r^2: '+ str(metrics.r2_score(valenceYTest10, valencePredicts10))+'\n')
    print('\n')

    arousalClassifier11 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier11 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier11.fit(channelPowersTrain11, arousalYTrain11)
    valenceClassifier11.fit(channelPowersTrain11, valenceYTrain11)
    arousalPredicts11 = arousalClassifier11.predict(channelPowersTest11)
    valencePredicts11 = valenceClassifier11.predict(channelPowersTest11)
    print('Arousal11 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest11, arousalPredicts11))+'\n')
    print('Arousal11 r^2: '+ str(metrics.r2_score(arousalYTest11, arousalPredicts11))+'\n')
    print('Valence11 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest11, valencePredicts11))+'\n')
    print('Valence11 r^2: '+ str(metrics.r2_score(valenceYTest11, valencePredicts11))+'\n')
    print('\n')

    arousalClassifier12 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier12 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier12.fit(channelPowersTrain12, arousalYTrain12)
    valenceClassifier12.fit(channelPowersTrain12, valenceYTrain12)
    arousalPredicts12 = arousalClassifier12.predict(channelPowersTest12)
    valencePredicts12 = valenceClassifier12.predict(channelPowersTest12)
    print('Arousal12 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest12, arousalPredicts12))+'\n')
    print('Arousal12 r^2: '+ str(metrics.r2_score(arousalYTest12, arousalPredicts12))+'\n')
    print('Valence12 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest12, valencePredicts12))+'\n')
    print('Valence12 r^2: '+ str(metrics.r2_score(valenceYTest12, valencePredicts12))+'\n')
    print('\n')

    arousalClassifier13 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier13 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier13.fit(channelPowersTrain13, arousalYTrain13)
    valenceClassifier13.fit(channelPowersTrain13, valenceYTrain13)
    arousalPredicts13 = arousalClassifier13.predict(channelPowersTest13)
    valencePredicts13 = valenceClassifier13.predict(channelPowersTest13)
    print('Arousal13 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest13, arousalPredicts13))+'\n')
    print('Arousal13 r^2: '+ str(metrics.r2_score(arousalYTest13, arousalPredicts13))+'\n')
    print('Valence13 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest13, valencePredicts13))+'\n')
    print('Valence13 r^2: '+ str(metrics.r2_score(valenceYTest13, valencePredicts13))+'\n')
    print('\n')

    arousalClassifier14 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier14 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier14.fit(channelPowersTrain14, arousalYTrain14)
    valenceClassifier14.fit(channelPowersTrain14, valenceYTrain14)
    arousalPredicts14 = arousalClassifier14.predict(channelPowersTest14)
    valencePredicts14 = valenceClassifier14.predict(channelPowersTest14)
    print('Arousal14 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest14, arousalPredicts14))+'\n')
    print('Arousal14 r^2: '+ str(metrics.r2_score(arousalYTest14, arousalPredicts14))+'\n')
    print('Valence14 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest14, valencePredicts14))+'\n')
    print('Valence14 r^2: '+ str(metrics.r2_score(valenceYTest14, valencePredicts14))+'\n')
    print('\n')

   #end
    print('Done')

if __name__ == '__main__':
    main()