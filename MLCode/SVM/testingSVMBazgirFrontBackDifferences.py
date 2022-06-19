import pickle
from numpy import tri #read in data
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
    subjectPowers = {}
    for subjectNumber in range(1,33):
        subjectPowerFile = 'DEAPBazgir\DEAPBazgir'+str(subjectNumber)+'.pickle'
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

    #separate into channels, pull channel out from each trial for each subject, have corresponding label dict to go off of?
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

    # take Front-Back differences
    # 0 = AF4-O2 (13-7), 1 = AF3-O1 (0-6), 2 = F8-T8 (12-9), 3 = F7-T7 (1-4), 4 = FC6-P8 (10-8), 5 = FC5-P7 (3-5), 6 = F4-02 (11-7), 7 = F3-O1 (2-6)
    channelPowersDifferences = {0:[],1:[],2:[],3:[],4:[],5:[],6:[], 7:[]}
    channelPowersLabels = {0:channelLabels[0],1:channelLabels[1],2:channelLabels[2],3:channelLabels[3],4:channelLabels[4],5:channelLabels[5],6:channelLabels[6],7:channelLabels[7]}
    for trial in range(1280):
        channelPowersDifferences[0].append([channelPowers[13][trial][x]-channelPowers[7][trial][x] for x in range(len(channelPowers[13][trial]))])
        channelPowersDifferences[1].append([channelPowers[0][trial][x]-channelPowers[6][trial][x] for x in range(len(channelPowers[0][trial]))])
        channelPowersDifferences[2].append([channelPowers[12][trial][x]-channelPowers[9][trial][x] for x in range(len(channelPowers[12][trial]))])
        channelPowersDifferences[3].append([channelPowers[1][trial][x]-channelPowers[4][trial][x] for x in range(len(channelPowers[1][trial]))])
        channelPowersDifferences[4].append([channelPowers[10][trial][x]-channelPowers[8][trial][x] for x in range(len(channelPowers[10][trial]))])
        channelPowersDifferences[5].append([channelPowers[3][trial][x]-channelPowers[5][trial][x] for x in range(len(channelPowers[3][trial]))])
        channelPowersDifferences[6].append([channelPowers[11][trial][x]-channelPowers[7][trial][x] for x in range(len(channelPowers[11][trial]))])
        channelPowersDifferences[7].append([channelPowers[2][trial][x]-channelPowers[6][trial][x] for x in range(len(channelPowers[2][trial]))])

    #duplicate to train on valence, arousal (for each make separate training labels (just arrays of the singular valence/arousal #))
    # THE Xs ARE THE SAME FOR VALENCE AND AROUSAL, SO NO NEED TO DUPLICATE
    # ONLY NEED TO ALTER THE Y-LISTS SO THAT THEY ARE BINARY
    arousalY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    valenceY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    for channelInd in range(8):
        channelLabel = channelPowersLabels[channelInd]
        for trialNum in range(1280):
            trialLabel = channelLabel[trialNum]
            if trialLabel[0] > 5:
                valenceY[channelInd].append(1)
            else:
                valenceY[channelInd].append(-1)
            if trialLabel[1] > 5:
                arousalY[channelInd].append(1)
            else:
                arousalY[channelInd].append(-1)

    #split data into test/train sets for each SVM
    print('Finished processing, splitting')
    channel0ArousalXTrain, channel0ArousalXTest, channel0ArousalYTrain, channel0ArousalYTest = train_test_split(channelPowersDifferences[0], arousalY[0], test_size = 0.2, random_state=42)
    channel0ValenceXTrain, channel0ValenceXTest, channel0ValenceYTrain, channel0ValenceYTest = train_test_split(channelPowersDifferences[0], valenceY[0], test_size = 0.2, random_state=42)
    
    channel1ArousalXTrain, channel1ArousalXTest, channel1ArousalYTrain, channel1ArousalYTest = train_test_split(channelPowersDifferences[1], arousalY[1], test_size = 0.2, random_state=42)
    channel1ValenceXTrain, channel1ValenceXTest, channel1ValenceYTrain, channel1ValenceYTest = train_test_split(channelPowersDifferences[1], valenceY[1], test_size = 0.2, random_state=42)

    channel2ArousalXTrain, channel2ArousalXTest, channel2ArousalYTrain, channel2ArousalYTest = train_test_split(channelPowersDifferences[2], arousalY[2], test_size = 0.2, random_state=42)
    channel2ValenceXTrain, channel2ValenceXTest, channel2ValenceYTrain, channel2ValenceYTest = train_test_split(channelPowersDifferences[2], valenceY[2], test_size = 0.2, random_state=42)
    
    channel3ArousalXTrain, channel3ArousalXTest, channel3ArousalYTrain, channel3ArousalYTest = train_test_split(channelPowersDifferences[3], arousalY[3], test_size = 0.2, random_state=42)
    channel3ValenceXTrain, channel3ValenceXTest, channel3ValenceYTrain, channel3ValenceYTest = train_test_split(channelPowersDifferences[3], valenceY[3], test_size = 0.2, random_state=42)

    channel4ArousalXTrain, channel4ArousalXTest, channel4ArousalYTrain, channel4ArousalYTest = train_test_split(channelPowersDifferences[4], arousalY[4], test_size = 0.2, random_state=42)
    channel4ValenceXTrain, channel4ValenceXTest, channel4ValenceYTrain, channel4ValenceYTest = train_test_split(channelPowersDifferences[4], valenceY[4], test_size = 0.2, random_state=42)

    channel5ArousalXTrain, channel5ArousalXTest, channel5ArousalYTrain, channel5ArousalYTest = train_test_split(channelPowersDifferences[5], arousalY[5], test_size = 0.2, random_state=42)
    channel5ValenceXTrain, channel5ValenceXTest, channel5ValenceYTrain, channel5ValenceYTest = train_test_split(channelPowersDifferences[5], valenceY[5], test_size = 0.2, random_state=42)

    channel6ArousalXTrain, channel6ArousalXTest, channel6ArousalYTrain, channel6ArousalYTest = train_test_split(channelPowersDifferences[6], arousalY[6], test_size = 0.2, random_state=42)
    channel6ValenceXTrain, channel6ValenceXTest, channel6ValenceYTrain, channel6ValenceYTest = train_test_split(channelPowersDifferences[6], valenceY[6], test_size = 0.2, random_state=42)

    channel7ArousalXTrain, channel7ArousalXTest, channel7ArousalYTrain, channel7ArousalYTest = train_test_split(channelPowersDifferences[7], arousalY[7], test_size = 0.2, random_state=42)
    channel7ValenceXTrain, channel7ValenceXTest, channel7ValenceYTrain, channel7ValenceYTest = train_test_split(channelPowersDifferences[7], valenceY[7], test_size = 0.2, random_state=42)

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

    #End
    print('Done')

if __name__=='__main__':
    main()