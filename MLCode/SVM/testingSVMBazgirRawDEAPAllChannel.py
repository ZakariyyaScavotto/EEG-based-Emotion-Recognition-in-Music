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

    allChannelPowers = [item for item in channelPowers.values()]
    newAllChannelPowers = []
    for channel in allChannelPowers:
        for item in channel:
            newAllChannelPowers.append(item)
    allArousalY = [rating for rating in arousalY.values()]
    newAllArousalY = []
    for channel in allArousalY:
        for item in channel:
            newAllArousalY.append(item)
    allValenceY = [rating for rating in valenceY.values()]
    newAllValenceY = []
    for channel in allValenceY:
        for item in channel:
            newAllValenceY.append(item)
    #split data into test/train sets for each SVM
    print('Finished processing, splitting')
    channel0ArousalXTrain, channel0ArousalXTest, channel0ArousalYTrain, channel0ArousalYTest = train_test_split(newAllChannelPowers, newAllArousalY, test_size = 0.2, random_state=42)
    channel0ValenceXTrain, channel0ValenceXTest, channel0ValenceYTrain, channel0ValenceYTest = train_test_split(newAllChannelPowers, newAllValenceY, test_size = 0.2, random_state=42)
    
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

    
    #End
    print('Done')

if __name__=='__main__':
    main()