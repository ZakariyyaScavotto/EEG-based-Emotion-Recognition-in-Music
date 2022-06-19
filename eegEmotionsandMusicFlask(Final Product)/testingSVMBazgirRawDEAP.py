import pickle #read in data
from sklearn.model_selection import train_test_split #used to split test data
from sklearn import svm #used to make svm
from sklearn import metrics #used to assess
from sklearn.model_selection import GridSearchCV

#Notes on the format of the DEAP data
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
    #read in DE into a dictionary subjectPowers[subject#] = [list of powers for all trials]
    subjectPowersFinal = {}
    for subjectNumber in range(1,24):
        subjectPowerFile = 'RawDEAPBazgir\RAWDEAPBazgir'+str(subjectNumber)+'.pickle'
        with open(subjectPowerFile, 'rb') as f:
            tempSubjectPowers = pickle.load(f)
        subjectPowersFinal[subjectNumber] = tempSubjectPowers

    #read in labels
    subjectLabels = {}
    for subjectNumber in range(1,24):
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
    for subjectNumber in range(1,24):
        subjectList = subjectPowersFinal[subjectNumber]
        count = 0
        for trial in subjectList:
            for channelInd in range(14):
                trialFeatures = trial[channelInd]
                unzipped = []
                for window in trialFeatures:
                    for element in window:
                        unzipped.append(element)
                channelPowers[channelInd].append(unzipped)
                channelLabels[channelInd].append(subjectLabels[subjectNumber][count])
            count += 1

    # revise lengths
    for channel in range(14):
        newChannelList = []
        channelList = channelPowers[channel]
        smallestLength = len(min(channelList, key=len))
        for item in channelList:
            newChannelList.append(item[:smallestLength])
        channelPowers[channel] = newChannelList
    # length checks:
    # for channel in range(14):
    #     channelList = channelPowers[channel]
    #     lenFirst = len(channelList[0])
    #     print(all(len(i)==lenFirst for i in channelList))
    #duplicate to train on valence, arousal (for each make separate training labels (just arrays of the singular valence/arousal #))
    # THE Xs ARE THE SAME FOR VALENCE AND AROUSAL, SO NO NEED TO DUPLICATE
    # ONLY NEED TO ALTER THE Y-LISTS SO THAT THEY ARE BINARY
    arousalY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    valenceY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    for channelInd in range(14):
        channelLabel = channelLabels[channelInd]
        for trialNum in range(len(channelLabel)):
            trialLabel = channelLabel[trialNum]
            if trialLabel[0] > 4.5:
                valenceY[channelInd].append(1)
            else:
                valenceY[channelInd].append(-1)
            if trialLabel[1] > 4.5:
                arousalY[channelInd].append(1)
            else:
                arousalY[channelInd].append(-1)

    #split data into test/train sets for each SVM
    print('Finished processing, splitting')
    channel0ArousalXTrain, channel0ArousalXTest, channel0ArousalYTrain, channel0ArousalYTest = train_test_split(channelPowers[0], arousalY[0], test_size = 0.2, random_state=42)
    channel0ValenceXTrain, channel0ValenceXTest, channel0ValenceYTrain, channel0ValenceYTest = train_test_split(channelPowers[0], valenceY[0], test_size = 0.2, random_state=42)
    
    channel1ArousalXTrain, channel1ArousalXTest, channel1ArousalYTrain, channel1ArousalYTest = train_test_split(channelPowers[1], arousalY[1], test_size = 0.2, random_state=42)
    channel1ValenceXTrain, channel1ValenceXTest, channel1ValenceYTrain, channel1ValenceYTest = train_test_split(channelPowers[1], valenceY[1], test_size = 0.2, random_state=42)

    channel2ArousalXTrain, channel2ArousalXTest, channel2ArousalYTrain, channel2ArousalYTest = train_test_split(channelPowers[2], arousalY[2], test_size = 0.2, random_state=42)
    channel2ValenceXTrain, channel2ValenceXTest, channel2ValenceYTrain, channel2ValenceYTest = train_test_split(channelPowers[2], valenceY[2], test_size = 0.2, random_state=42)
    
    channel3ArousalXTrain, channel3ArousalXTest, channel3ArousalYTrain, channel3ArousalYTest = train_test_split(channelPowers[3], arousalY[3], test_size = 0.2, random_state=42)
    channel3ValenceXTrain, channel3ValenceXTest, channel3ValenceYTrain, channel3ValenceYTest = train_test_split(channelPowers[3], valenceY[3], test_size = 0.2, random_state=42)

    channel4ArousalXTrain, channel4ArousalXTest, channel4ArousalYTrain, channel4ArousalYTest = train_test_split(channelPowers[4], arousalY[4], test_size = 0.2, random_state=42)
    channel4ValenceXTrain, channel4ValenceXTest, channel4ValenceYTrain, channel4ValenceYTest = train_test_split(channelPowers[4], valenceY[4], test_size = 0.2, random_state=42)

    channel5ArousalXTrain, channel5ArousalXTest, channel5ArousalYTrain, channel5ArousalYTest = train_test_split(channelPowers[5], arousalY[5], test_size = 0.2, random_state=42)
    channel5ValenceXTrain, channel5ValenceXTest, channel5ValenceYTrain, channel5ValenceYTest = train_test_split(channelPowers[5], valenceY[5], test_size = 0.2, random_state=42)

    channel6ArousalXTrain, channel6ArousalXTest, channel6ArousalYTrain, channel6ArousalYTest = train_test_split(channelPowers[6], arousalY[6], test_size = 0.2, random_state=42)
    channel6ValenceXTrain, channel6ValenceXTest, channel6ValenceYTrain, channel6ValenceYTest = train_test_split(channelPowers[6], valenceY[6], test_size = 0.2, random_state=42)

    channel7ArousalXTrain, channel7ArousalXTest, channel7ArousalYTrain, channel7ArousalYTest = train_test_split(channelPowers[7], arousalY[7], test_size = 0.2, random_state=42)
    channel7ValenceXTrain, channel7ValenceXTest, channel7ValenceYTrain, channel7ValenceYTest = train_test_split(channelPowers[7], valenceY[7], test_size = 0.2, random_state=42)

    channel8ArousalXTrain, channel8ArousalXTest, channel8ArousalYTrain, channel8ArousalYTest = train_test_split(channelPowers[8], arousalY[8], test_size = 0.2, random_state=42)
    channel8ValenceXTrain, channel8ValenceXTest, channel8ValenceYTrain, channel8ValenceYTest = train_test_split(channelPowers[8], valenceY[8], test_size = 0.2, random_state=42)

    channel9ArousalXTrain, channel9ArousalXTest, channel9ArousalYTrain, channel9ArousalYTest = train_test_split(channelPowers[9], arousalY[9], test_size = 0.2, random_state=42)
    channel9ValenceXTrain, channel9ValenceXTest, channel9ValenceYTrain, channel9ValenceYTest = train_test_split(channelPowers[9], valenceY[9], test_size = 0.2, random_state=42)

    channel10ArousalXTrain, channel10ArousalXTest, channel10ArousalYTrain, channel10ArousalYTest = train_test_split(channelPowers[10], arousalY[10], test_size = 0.2, random_state=42)
    channel10ValenceXTrain, channel10ValenceXTest, channel10ValenceYTrain, channel10ValenceYTest = train_test_split(channelPowers[10], valenceY[10], test_size = 0.2, random_state=42)

    channel11ArousalXTrain, channel11ArousalXTest, channel11ArousalYTrain, channel11ArousalYTest = train_test_split(channelPowers[11], arousalY[11], test_size = 0.2, random_state=42)
    channel11ValenceXTrain, channel11ValenceXTest, channel11ValenceYTrain, channel11ValenceYTest = train_test_split(channelPowers[11], valenceY[11], test_size = 0.2, random_state=42)

    channel12ArousalXTrain, channel12ArousalXTest, channel12ArousalYTrain, channel12ArousalYTest = train_test_split(channelPowers[12], arousalY[12], test_size = 0.2, random_state=42)
    channel12ValenceXTrain, channel12ValenceXTest, channel12ValenceYTrain, channel12ValenceYTest = train_test_split(channelPowers[12], valenceY[12], test_size = 0.2, random_state=42)

    channel13ArousalXTrain, channel13ArousalXTest, channel13ArousalYTrain, channel13ArousalYTest = train_test_split(channelPowers[13], arousalY[13], test_size = 0.2, random_state=42)
    channel13ValenceXTrain, channel13ValenceXTest, channel13ValenceYTrain, channel13ValenceYTest = train_test_split(channelPowers[13], valenceY[13], test_size = 0.2, random_state=42)

    #train each SVM
    print('Starting training all SVMs')
    paramGrid = {'C': [0.1,0.5,1, 10, 100], 'gamma': [1,0.1,0.05,0.01,0.001],'kernel': ['rbf']}

    channel0ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel0ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel0ArousalGrid.fit(channel0ArousalXTrain, channel0ArousalYTrain)
    channel0ValenceGrid.fit(channel0ValenceXTrain, channel0ValenceYTrain)
    print('grid channel0Arousal results')
    print(channel0ArousalGrid.best_estimator_)
    print('grid channel0Valence results')
    print(channel0ValenceGrid.best_estimator_)
    arousalPredicts = channel0ArousalGrid.predict(channel0ArousalXTest)
    valencePredicts = channel0ValenceGrid.predict(channel0ValenceXTest)
    print('channel0Arousal Accuracy: ',metrics.accuracy_score(channel0ArousalYTest, arousalPredicts))
    print('channel0Valence Accuracy: ',metrics.accuracy_score(channel0ValenceYTest, valencePredicts))

    channel1ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel1ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel1ArousalGrid.fit(channel1ArousalXTrain, channel1ArousalYTrain)
    channel1ValenceGrid.fit(channel1ValenceXTrain, channel1ValenceYTrain)
    print('grid channel1Arousal results')
    print(channel1ArousalGrid.best_estimator_)
    print('grid channel1Valence results')
    print(channel1ValenceGrid.best_estimator_)
    arousalPredicts = channel1ArousalGrid.predict(channel1ArousalXTest)
    valencePredicts = channel1ValenceGrid.predict(channel1ValenceXTest)
    print('channel1Arousal Accuracy: ',metrics.accuracy_score(channel1ArousalYTest, arousalPredicts))
    print('channel1Valence Accuracy: ',metrics.accuracy_score(channel1ValenceYTest, valencePredicts))

    channel2ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel2ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel2ArousalGrid.fit(channel2ArousalXTrain, channel2ArousalYTrain)
    channel2ValenceGrid.fit(channel2ValenceXTrain, channel2ValenceYTrain)
    print('grid channel2Arousal results')
    print(channel2ArousalGrid.best_estimator_)
    print('grid channel2Valence results')
    print(channel2ValenceGrid.best_estimator_)
    arousalPredicts = channel2ArousalGrid.predict(channel2ArousalXTest)
    valencePredicts = channel2ValenceGrid.predict(channel2ValenceXTest)
    print('channel2Arousal Accuracy: ',metrics.accuracy_score(channel2ArousalYTest, arousalPredicts))
    print('channel2Valence Accuracy: ',metrics.accuracy_score(channel2ValenceYTest, valencePredicts))

    channel3ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel3ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel3ArousalGrid.fit(channel3ArousalXTrain, channel3ArousalYTrain)
    channel3ValenceGrid.fit(channel3ValenceXTrain, channel3ValenceYTrain)
    print('grid channel3Arousal results')
    print(channel3ArousalGrid.best_estimator_)
    print('grid channel3Valence results')
    print(channel3ValenceGrid.best_estimator_)
    arousalPredicts = channel3ArousalGrid.predict(channel3ArousalXTest)
    valencePredicts = channel3ValenceGrid.predict(channel3ValenceXTest)
    print('channel3Arousal Accuracy: ',metrics.accuracy_score(channel3ArousalYTest, arousalPredicts))
    print('channel3Valence Accuracy: ',metrics.accuracy_score(channel3ValenceYTest, valencePredicts))

    channel4ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel4ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel4ArousalGrid.fit(channel4ArousalXTrain, channel4ArousalYTrain)
    channel4ValenceGrid.fit(channel4ValenceXTrain, channel4ValenceYTrain)
    print('grid channel4Arousal results')
    print(channel4ArousalGrid.best_estimator_)
    print('grid channel4Valence results')
    print(channel4ValenceGrid.best_estimator_)
    arousalPredicts = channel4ArousalGrid.predict(channel4ArousalXTest)
    valencePredicts = channel4ValenceGrid.predict(channel4ValenceXTest)
    print('channel4Arousal Accuracy: ',metrics.accuracy_score(channel4ArousalYTest, arousalPredicts))
    print('channel4Valence Accuracy: ',metrics.accuracy_score(channel4ValenceYTest, valencePredicts))

    channel5ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel5ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel5ArousalGrid.fit(channel5ArousalXTrain, channel5ArousalYTrain)
    channel5ValenceGrid.fit(channel5ValenceXTrain, channel5ValenceYTrain)
    print('grid channel5Arousal results')
    print(channel5ArousalGrid.best_estimator_)
    print('grid channel5Valence results')
    print(channel5ValenceGrid.best_estimator_)
    arousalPredicts = channel5ArousalGrid.predict(channel5ArousalXTest)
    valencePredicts = channel5ValenceGrid.predict(channel5ValenceXTest)
    print('channel5Arousal Accuracy: ',metrics.accuracy_score(channel5ArousalYTest, arousalPredicts))
    print('channel5Valence Accuracy: ',metrics.accuracy_score(channel5ValenceYTest, valencePredicts))

    channel6ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel6ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel6ArousalGrid.fit(channel6ArousalXTrain, channel6ArousalYTrain)
    channel6ValenceGrid.fit(channel6ValenceXTrain, channel6ValenceYTrain)
    print('grid channel6Arousal results')
    print(channel6ArousalGrid.best_estimator_)
    print('grid channel6Valence results')
    print(channel6ValenceGrid.best_estimator_)
    arousalPredicts = channel6ArousalGrid.predict(channel6ArousalXTest)
    valencePredicts = channel6ValenceGrid.predict(channel6ValenceXTest)
    print('channel6Arousal Accuracy: ',metrics.accuracy_score(channel6ArousalYTest, arousalPredicts))
    print('channel6Valence Accuracy: ',metrics.accuracy_score(channel6ValenceYTest, valencePredicts))

    channel7ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel7ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel7ArousalGrid.fit(channel7ArousalXTrain, channel7ArousalYTrain)
    channel7ValenceGrid.fit(channel7ValenceXTrain, channel7ValenceYTrain)
    print('grid channel7Arousal results')
    print(channel7ArousalGrid.best_estimator_)
    print('grid channel7Valence results')
    print(channel7ValenceGrid.best_estimator_)
    arousalPredicts = channel7ArousalGrid.predict(channel7ArousalXTest)
    valencePredicts = channel7ValenceGrid.predict(channel7ValenceXTest)
    print('channel7Arousal Accuracy: ',metrics.accuracy_score(channel7ArousalYTest, arousalPredicts))
    print('channel7Valence Accuracy: ',metrics.accuracy_score(channel7ValenceYTest, valencePredicts))

    channel8ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel8ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel8ArousalGrid.fit(channel8ArousalXTrain, channel8ArousalYTrain)
    channel8ValenceGrid.fit(channel8ValenceXTrain, channel8ValenceYTrain)
    print('grid channel8Arousal results')
    print(channel8ArousalGrid.best_estimator_)
    print('grid channel8Valence results')
    print(channel8ValenceGrid.best_estimator_)
    arousalPredicts = channel8ArousalGrid.predict(channel8ArousalXTest)
    valencePredicts = channel8ValenceGrid.predict(channel8ValenceXTest)
    print('channel8Arousal Accuracy: ',metrics.accuracy_score(channel8ArousalYTest, arousalPredicts))
    print('channel8Valence Accuracy: ',metrics.accuracy_score(channel8ValenceYTest, valencePredicts))

    channel9ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel9ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel9ArousalGrid.fit(channel9ArousalXTrain, channel9ArousalYTrain)
    channel9ValenceGrid.fit(channel9ValenceXTrain, channel9ValenceYTrain)
    print('grid channel9Arousal results')
    print(channel9ArousalGrid.best_estimator_)
    print('grid channel9Valence results')
    print(channel9ValenceGrid.best_estimator_)
    arousalPredicts = channel9ArousalGrid.predict(channel9ArousalXTest)
    valencePredicts = channel9ValenceGrid.predict(channel9ValenceXTest)
    print('channel9Arousal Accuracy: ',metrics.accuracy_score(channel9ArousalYTest, arousalPredicts))
    print('channel9Valence Accuracy: ',metrics.accuracy_score(channel9ValenceYTest, valencePredicts))

    channel10ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel10ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel10ArousalGrid.fit(channel10ArousalXTrain, channel10ArousalYTrain)
    channel10ValenceGrid.fit(channel10ValenceXTrain, channel10ValenceYTrain)
    print('grid channel10Arousal results')
    print(channel10ArousalGrid.best_estimator_)
    print('grid channel10Valence results')
    print(channel10ValenceGrid.best_estimator_)
    arousalPredicts = channel10ArousalGrid.predict(channel10ArousalXTest)
    valencePredicts = channel10ValenceGrid.predict(channel10ValenceXTest)
    print('channel10Arousal Accuracy: ',metrics.accuracy_score(channel10ArousalYTest, arousalPredicts))
    print('channel10Valence Accuracy: ',metrics.accuracy_score(channel10ValenceYTest, valencePredicts))

    channel11ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel11ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel11ArousalGrid.fit(channel11ArousalXTrain, channel11ArousalYTrain)
    channel11ValenceGrid.fit(channel11ValenceXTrain, channel11ValenceYTrain)
    print('grid channel11Arousal results')
    print(channel11ArousalGrid.best_estimator_)
    print('grid channel11Valence results')
    print(channel11ValenceGrid.best_estimator_)
    arousalPredicts = channel11ArousalGrid.predict(channel11ArousalXTest)
    valencePredicts = channel11ValenceGrid.predict(channel11ValenceXTest)
    print('channel11Arousal Accuracy: ',metrics.accuracy_score(channel11ArousalYTest, arousalPredicts))
    print('channel11Valence Accuracy: ',metrics.accuracy_score(channel11ValenceYTest, valencePredicts))

    channel12ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel12ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel12ArousalGrid.fit(channel12ArousalXTrain, channel12ArousalYTrain)
    channel12ValenceGrid.fit(channel12ValenceXTrain, channel12ValenceYTrain)
    print('grid channel12Arousal results')
    print(channel12ArousalGrid.best_estimator_)
    print('grid channel12Valence results')
    print(channel12ValenceGrid.best_estimator_)
    arousalPredicts = channel12ArousalGrid.predict(channel12ArousalXTest)
    valencePredicts = channel12ValenceGrid.predict(channel12ValenceXTest)
    print('channel12Arousal Accuracy: ',metrics.accuracy_score(channel12ArousalYTest, arousalPredicts))
    print('channel12Valence Accuracy: ',metrics.accuracy_score(channel12ValenceYTest, valencePredicts))

    channel13ArousalGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel13ValenceGrid = GridSearchCV(svm.SVC(), paramGrid, refit=True, verbose=0)
    channel13ArousalGrid.fit(channel13ArousalXTrain, channel13ArousalYTrain)
    channel13ValenceGrid.fit(channel13ValenceXTrain, channel13ValenceYTrain)
    print('grid channel13Arousal results')
    print(channel13ArousalGrid.best_estimator_)
    print('grid channel13Valence results')
    print(channel13ValenceGrid.best_estimator_)
    arousalPredicts = channel13ArousalGrid.predict(channel13ArousalXTest)
    valencePredicts = channel13ValenceGrid.predict(channel13ValenceXTest)
    print('channel13Arousal Accuracy: ',metrics.accuracy_score(channel13ArousalYTest, arousalPredicts))
    print('channel13Valence Accuracy: ',metrics.accuracy_score(channel13ValenceYTest, valencePredicts))

    #End
    print('Done')

if __name__=='__main__':
    main()