import pickle #read in data
from sklearn.model_selection import train_test_split #used to split test data
from sklearn import svm #used to make svm
from sklearn import metrics #used to assess

#Notes on the format of the DEAP data
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def main():
  #SVM For all channels  
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
    
   #train each SVM
    arousalCLF = svm.SVC(kernel='rbf', C=100, gamma=0.01)
    valenceCLF = svm.SVC(kernel='rbf', C=10, gamma=0.1)
    print('Starting Training')
    arousalCLF.fit(arousalXTrain, arousalYTrain)
    print('Finished training arousal')
    valenceCLF.fit(valenceXTrain, valenceYTrain)
    print('Finished training valence')
    
   #look at accuracies
    arousalPredicts = arousalCLF.predict(arousalXTest)
    valencePredicts = valenceCLF.predict(valenceXTest)
    print('All channel arousal Accuracy: ',metrics.accuracy_score(arousalYTest, arousalPredicts))
    print('All channel valence Accuracy: ',metrics.accuracy_score(valenceYTest, valencePredicts))
    
   #End
    print('All channels done')
    print()

  #SVM for individual channels
    channelList = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
       
    for channelInd in range(32): #32 channels
       #collect data per channel
        
        arousalX, valenceX = [],[]
        for subj in range(1,33): #32 subjects
            for trial in range(40): 
                arousalX.append([subjectPowers[subj][trial][channelInd]])
                valenceX.append([subjectPowers[subj][trial][channelInd]])
        arousalY, valenceY = [], []
        for subj in range(1,33):
            for trial in range(40):
                trialLabel = subjectLabels[subj][trial]
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
        
       #train each SVM
        arousalCLF = svm.SVC(kernel='rbf', C=100, gamma=0.01)
        valenceCLF = svm.SVC(kernel='rbf', C=10, gamma=0.1)
        #print('Starting Training')
        arousalCLF.fit(arousalXTrain, arousalYTrain)
        #print('Finished training arousal')
        valenceCLF.fit(valenceXTrain, valenceYTrain)
        #print('Finished training valence')
        
       #look at accuracies
        arousalPredicts = arousalCLF.predict(arousalXTest)
        valencePredicts = valenceCLF.predict(valenceXTest)
        print(channelList[channelInd]+' arousal Accuracy: ',metrics.accuracy_score(arousalYTest, arousalPredicts))
        print(channelList[channelInd]+' valence Accuracy: ',metrics.accuracy_score(valenceYTest, valencePredicts))
        print()
    
    print('End of Program')

if __name__=='__main__':
    main()