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
   #read in DE into a dictionary subjectPowers[subject#] = [list of DE]
    subjectPowers, tempSubjectPowersDict = {}, {}
    fullDEList = []
    for subjectNumber in range(1,33):
        subjectPowerFile = 'DEAPDiffEntropySUBBANDS\DEAPDiffEntropySUBBANDSSubject'+str(subjectNumber)+'.pickle'
        with open(subjectPowerFile, 'rb') as f:
            tempSubjectPowers = pickle.load(f)
        
        #v1 scaling: mean and std. dev per trial
        fullSubjectDE = []
        for trial in tempSubjectPowers:
            meanDE, stdDevDE = statistics.mean(trial), statistics.pstdev(trial)
            fullSubjectDE.append([(de - meanDE)/stdDevDE + 0.5 for de in trial])
        subjectPowers[subjectNumber] = fullSubjectDE
        
        #v2 scaling: mean and std. dev across all trials per subject
        # fullSubjectDE, originalTrialsList, everyDEList = [],[], []
        # for trial in tempSubjectPowers:
        #     originalTrialsList.append(trial)
        #     for de in trial:
        #         everyDEList.append(de)
        # meanSubject, stdDevSubject = statistics.mean(everyDEList), statistics.pstdev(everyDEList)
        # for trial in tempSubjectPowers:
        #     fullSubjectDE.append([(de - meanSubject)/stdDevSubject + 0.5 for de in trial])
        # subjectPowers[subjectNumber] = fullSubjectDE

    #v3 scaling: mean and std. dev across all subjects
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
    
   #duplicate to train on valence, arousal FOR EACH CHANNEL
    arousalX, arousalY, valenceX, valenceY = {}, {}, {}, {}
    for subjectNumber in range(1,33):
        subjectPow, subjectLabel = subjectPowers[subjectNumber],subjectLabels[subjectNumber]
        tempsubjectTrials, tempsubjectLabels = [], [] 
        for trialNumber in range(40):
            trialPower, trialLabel = np.array(subjectPow[trialNumber]), subjectLabel[trialNumber]
            tempsubjectTrials.append(trialPower)
            tempsubjectLabels.append(trialLabel)
        arousalX[subjectNumber] = tempsubjectTrials
        arousalY[subjectNumber] = tempsubjectLabels
        valenceX[subjectNumber] = tempsubjectTrials
        valenceY[subjectNumber] = tempsubjectLabels
    # clean out labels
    for subjectNumber in range(1,33):
        tempArousal, tempValence = [],[]
        for labelList in arousalY[subjectNumber]:
            tempArousal.append(labelList[0])
        for labelList2 in valenceY[subjectNumber]:
            tempValence.append(labelList2[1])
        arousalY[subjectNumber] = tempArousal
        valenceY[subjectNumber] = tempValence
    
   #split data into test/train sets for each DBN (and each channel), then convert to np arrays
    arousalXTrain1, arousalXTest1, arousalYTrain1, arousalYTest1 = train_test_split(arousalX[1], arousalY[1], test_size = 0.3, random_state=42)
    valenceXTrain1, valenceXTest1, valenceYTrain1, valenceYTest1 = train_test_split(valenceX[1], valenceY[1], test_size = 0.3, random_state=42)
    arousalXTrain1, arousalXTest1, arousalYTrain1, arousalYTest1, valenceXTrain1, valenceXTest1, valenceYTrain1, valenceYTest1 = np.array(arousalXTrain1), np.array(arousalXTest1), np.array(arousalYTrain1), np.array(arousalYTest1), np.array(valenceXTrain1), np.array(valenceXTest1), np.array(valenceYTrain1), np.array(valenceYTest1)

    arousalXTrain2, arousalXTest2, arousalYTrain2, arousalYTest2 = train_test_split(arousalX[2], arousalY[2], test_size = 0.3, random_state=42)
    valenceXTrain2, valenceXTest2, valenceYTrain2, valenceYTest2 = train_test_split(valenceX[2], valenceY[2], test_size = 0.3, random_state=42)
    arousalXTrain2, arousalXTest2, arousalYTrain2, arousalYTest2, valenceXTrain2, valenceXTest2, valenceYTrain2, valenceYTest2 = np.array(arousalXTrain2), np.array(arousalXTest2), np.array(arousalYTrain2), np.array(arousalYTest2), np.array(valenceXTrain2), np.array(valenceXTest2), np.array(valenceYTrain2), np.array(valenceYTest2)

    arousalXTrain3, arousalXTest3, arousalYTrain3, arousalYTest3 = train_test_split(arousalX[3], arousalY[3], test_size = 0.3, random_state=42)
    valenceXTrain3, valenceXTest3, valenceYTrain3, valenceYTest3 = train_test_split(valenceX[3], valenceY[3], test_size = 0.3, random_state=42)
    arousalXTrain3, arousalXTest3, arousalYTrain3, arousalYTest3, valenceXTrain3, valenceXTest3, valenceYTrain3, valenceYTest3 = np.array(arousalXTrain3), np.array(arousalXTest3), np.array(arousalYTrain3), np.array(arousalYTest3), np.array(valenceXTrain3), np.array(valenceXTest3), np.array(valenceYTrain3), np.array(valenceYTest3)

    arousalXTrain4, arousalXTest4, arousalYTrain4, arousalYTest4 = train_test_split(arousalX[4], arousalY[4], test_size = 0.3, random_state=42)
    valenceXTrain4, valenceXTest4, valenceYTrain4, valenceYTest4 = train_test_split(valenceX[4], valenceY[4], test_size = 0.3, random_state=42)
    arousalXTrain4, arousalXTest4, arousalYTrain4, arousalYTest4, valenceXTrain4, valenceXTest4, valenceYTrain4, valenceYTest4 = np.array(arousalXTrain4), np.array(arousalXTest4), np.array(arousalYTrain4), np.array(arousalYTest4), np.array(valenceXTrain4), np.array(valenceXTest4), np.array(valenceYTrain4), np.array(valenceYTest4)

    arousalXTrain5, arousalXTest5, arousalYTrain5, arousalYTest5 = train_test_split(arousalX[5], arousalY[5], test_size = 0.3, random_state=42)
    valenceXTrain5, valenceXTest5, valenceYTrain5, valenceYTest5 = train_test_split(valenceX[5], valenceY[5], test_size = 0.3, random_state=42)
    arousalXTrain5, arousalXTest5, arousalYTrain5, arousalYTest5, valenceXTrain5, valenceXTest5, valenceYTrain5, valenceYTest5 = np.array(arousalXTrain5), np.array(arousalXTest5), np.array(arousalYTrain5), np.array(arousalYTest5), np.array(valenceXTrain5), np.array(valenceXTest5), np.array(valenceYTrain5), np.array(valenceYTest5)

    arousalXTrain6, arousalXTest6, arousalYTrain6, arousalYTest6 = train_test_split(arousalX[6], arousalY[6], test_size = 0.3, random_state=42)
    valenceXTrain6, valenceXTest6, valenceYTrain6, valenceYTest6 = train_test_split(valenceX[6], valenceY[6], test_size = 0.3, random_state=42)
    arousalXTrain6, arousalXTest6, arousalYTrain6, arousalYTest6, valenceXTrain6, valenceXTest6, valenceYTrain6, valenceYTest6 = np.array(arousalXTrain6), np.array(arousalXTest6), np.array(arousalYTrain6), np.array(arousalYTest6), np.array(valenceXTrain6), np.array(valenceXTest6), np.array(valenceYTrain6), np.array(valenceYTest6)

    arousalXTrain7, arousalXTest7, arousalYTrain7, arousalYTest7 = train_test_split(arousalX[7], arousalY[7], test_size = 0.3, random_state=42)
    valenceXTrain7, valenceXTest7, valenceYTrain7, valenceYTest7 = train_test_split(valenceX[7], valenceY[7], test_size = 0.3, random_state=42)
    arousalXTrain7, arousalXTest7, arousalYTrain7, arousalYTest7, valenceXTrain7, valenceXTest7, valenceYTrain7, valenceYTest7 = np.array(arousalXTrain7), np.array(arousalXTest7), np.array(arousalYTrain7), np.array(arousalYTest7), np.array(valenceXTrain7), np.array(valenceXTest7), np.array(valenceYTrain7), np.array(valenceYTest7)

    arousalXTrain8, arousalXTest8, arousalYTrain8, arousalYTest8 = train_test_split(arousalX[8], arousalY[8], test_size = 0.3, random_state=42)
    valenceXTrain8, valenceXTest8, valenceYTrain8, valenceYTest8 = train_test_split(valenceX[8], valenceY[8], test_size = 0.3, random_state=42)
    arousalXTrain8, arousalXTest8, arousalYTrain8, arousalYTest8, valenceXTrain8, valenceXTest8, valenceYTrain8, valenceYTest8 = np.array(arousalXTrain8), np.array(arousalXTest8), np.array(arousalYTrain8), np.array(arousalYTest8), np.array(valenceXTrain8), np.array(valenceXTest8), np.array(valenceYTrain8), np.array(valenceYTest8)

    arousalXTrain9, arousalXTest9, arousalYTrain9, arousalYTest9 = train_test_split(arousalX[9], arousalY[9], test_size = 0.3, random_state=42)
    valenceXTrain9, valenceXTest9, valenceYTrain9, valenceYTest9 = train_test_split(valenceX[9], valenceY[9], test_size = 0.3, random_state=42)
    arousalXTrain9, arousalXTest9, arousalYTrain9, arousalYTest9, valenceXTrain9, valenceXTest9, valenceYTrain9, valenceYTest9 = np.array(arousalXTrain9), np.array(arousalXTest9), np.array(arousalYTrain9), np.array(arousalYTest9), np.array(valenceXTrain9), np.array(valenceXTest9), np.array(valenceYTrain9), np.array(valenceYTest9)

    arousalXTrain10, arousalXTest10, arousalYTrain10, arousalYTest10 = train_test_split(arousalX[10], arousalY[10], test_size = 0.3, random_state=42)
    valenceXTrain10, valenceXTest10, valenceYTrain10, valenceYTest10 = train_test_split(valenceX[10], valenceY[10], test_size = 0.3, random_state=42)
    arousalXTrain10, arousalXTest10, arousalYTrain10, arousalYTest10, valenceXTrain10, valenceXTest10, valenceYTrain10, valenceYTest10 = np.array(arousalXTrain10), np.array(arousalXTest10), np.array(arousalYTrain10), np.array(arousalYTest10), np.array(valenceXTrain10), np.array(valenceXTest10), np.array(valenceYTrain10), np.array(valenceYTest10)

    arousalXTrain11, arousalXTest11, arousalYTrain11, arousalYTest11 = train_test_split(arousalX[11], arousalY[11], test_size = 0.3, random_state=42)
    valenceXTrain11, valenceXTest11, valenceYTrain11, valenceYTest11 = train_test_split(valenceX[11], valenceY[11], test_size = 0.3, random_state=42)
    arousalXTrain11, arousalXTest11, arousalYTrain11, arousalYTest11, valenceXTrain11, valenceXTest11, valenceYTrain11, valenceYTest11 = np.array(arousalXTrain11), np.array(arousalXTest11), np.array(arousalYTrain11), np.array(arousalYTest11), np.array(valenceXTrain11), np.array(valenceXTest11), np.array(valenceYTrain11), np.array(valenceYTest11)

    arousalXTrain12, arousalXTest12, arousalYTrain12, arousalYTest12 = train_test_split(arousalX[12], arousalY[12], test_size = 0.3, random_state=42)
    valenceXTrain12, valenceXTest12, valenceYTrain12, valenceYTest12 = train_test_split(valenceX[12], valenceY[12], test_size = 0.3, random_state=42)
    arousalXTrain12, arousalXTest12, arousalYTrain12, arousalYTest12, valenceXTrain12, valenceXTest12, valenceYTrain12, valenceYTest12 = np.array(arousalXTrain12), np.array(arousalXTest12), np.array(arousalYTrain12), np.array(arousalYTest12), np.array(valenceXTrain12), np.array(valenceXTest12), np.array(valenceYTrain12), np.array(valenceYTest12)

    arousalXTrain13, arousalXTest13, arousalYTrain13, arousalYTest13 = train_test_split(arousalX[13], arousalY[13], test_size = 0.3, random_state=42)
    valenceXTrain13, valenceXTest13, valenceYTrain13, valenceYTest13 = train_test_split(valenceX[13], valenceY[13], test_size = 0.3, random_state=42)
    arousalXTrain13, arousalXTest13, arousalYTrain13, arousalYTest13, valenceXTrain13, valenceXTest13, valenceYTrain13, valenceYTest13 = np.array(arousalXTrain13), np.array(arousalXTest13), np.array(arousalYTrain13), np.array(arousalYTest13), np.array(valenceXTrain13), np.array(valenceXTest13), np.array(valenceYTrain13), np.array(valenceYTest13)

    arousalXTrain14, arousalXTest14, arousalYTrain14, arousalYTest14 = train_test_split(arousalX[14], arousalY[14], test_size = 0.3, random_state=42)
    valenceXTrain14, valenceXTest14, valenceYTrain14, valenceYTest14 = train_test_split(valenceX[14], valenceY[14], test_size = 0.3, random_state=42)
    arousalXTrain14, arousalXTest14, arousalYTrain14, arousalYTest14, valenceXTrain14, valenceXTest14, valenceYTrain14, valenceYTest14 = np.array(arousalXTrain14), np.array(arousalXTest14), np.array(arousalYTrain14), np.array(arousalYTest14), np.array(valenceXTrain14), np.array(valenceXTest14), np.array(valenceYTrain14), np.array(valenceYTest14)

    arousalXTrain15, arousalXTest15, arousalYTrain15, arousalYTest15 = train_test_split(arousalX[15], arousalY[15], test_size = 0.3, random_state=42)
    valenceXTrain15, valenceXTest15, valenceYTrain15, valenceYTest15 = train_test_split(valenceX[15], valenceY[15], test_size = 0.3, random_state=42)
    arousalXTrain15, arousalXTest15, arousalYTrain15, arousalYTest15, valenceXTrain15, valenceXTest15, valenceYTrain15, valenceYTest15 = np.array(arousalXTrain15), np.array(arousalXTest15), np.array(arousalYTrain15), np.array(arousalYTest15), np.array(valenceXTrain15), np.array(valenceXTest15), np.array(valenceYTrain15), np.array(valenceYTest15)

    arousalXTrain16, arousalXTest16, arousalYTrain16, arousalYTest16 = train_test_split(arousalX[16], arousalY[16], test_size = 0.3, random_state=42)
    valenceXTrain16, valenceXTest16, valenceYTrain16, valenceYTest16 = train_test_split(valenceX[16], valenceY[16], test_size = 0.3, random_state=42)
    arousalXTrain16, arousalXTest16, arousalYTrain16, arousalYTest16, valenceXTrain16, valenceXTest16, valenceYTrain16, valenceYTest16 = np.array(arousalXTrain16), np.array(arousalXTest16), np.array(arousalYTrain16), np.array(arousalYTest16), np.array(valenceXTrain16), np.array(valenceXTest16), np.array(valenceYTrain16), np.array(valenceYTest16)

    arousalXTrain17, arousalXTest17, arousalYTrain17, arousalYTest17 = train_test_split(arousalX[17], arousalY[17], test_size = 0.3, random_state=42)
    valenceXTrain17, valenceXTest17, valenceYTrain17, valenceYTest17 = train_test_split(valenceX[17], valenceY[17], test_size = 0.3, random_state=42)
    arousalXTrain17, arousalXTest17, arousalYTrain17, arousalYTest17, valenceXTrain17, valenceXTest17, valenceYTrain17, valenceYTest17 = np.array(arousalXTrain17), np.array(arousalXTest17), np.array(arousalYTrain17), np.array(arousalYTest17), np.array(valenceXTrain17), np.array(valenceXTest17), np.array(valenceYTrain17), np.array(valenceYTest17)

    arousalXTrain18, arousalXTest18, arousalYTrain18, arousalYTest18 = train_test_split(arousalX[18], arousalY[18], test_size = 0.3, random_state=42)
    valenceXTrain18, valenceXTest18, valenceYTrain18, valenceYTest18 = train_test_split(valenceX[18], valenceY[18], test_size = 0.3, random_state=42)
    arousalXTrain18, arousalXTest18, arousalYTrain18, arousalYTest18, valenceXTrain18, valenceXTest18, valenceYTrain18, valenceYTest18 = np.array(arousalXTrain18), np.array(arousalXTest18), np.array(arousalYTrain18), np.array(arousalYTest18), np.array(valenceXTrain18), np.array(valenceXTest18), np.array(valenceYTrain18), np.array(valenceYTest18)

    arousalXTrain19, arousalXTest19, arousalYTrain19, arousalYTest19 = train_test_split(arousalX[19], arousalY[19], test_size = 0.3, random_state=42)
    valenceXTrain19, valenceXTest19, valenceYTrain19, valenceYTest19 = train_test_split(valenceX[19], valenceY[19], test_size = 0.3, random_state=42)
    arousalXTrain19, arousalXTest19, arousalYTrain19, arousalYTest19, valenceXTrain19, valenceXTest19, valenceYTrain19, valenceYTest19 = np.array(arousalXTrain19), np.array(arousalXTest19), np.array(arousalYTrain19), np.array(arousalYTest19), np.array(valenceXTrain19), np.array(valenceXTest19), np.array(valenceYTrain19), np.array(valenceYTest19)

    arousalXTrain20, arousalXTest20, arousalYTrain20, arousalYTest20 = train_test_split(arousalX[20], arousalY[20], test_size = 0.3, random_state=42)
    valenceXTrain20, valenceXTest20, valenceYTrain20, valenceYTest20 = train_test_split(valenceX[20], valenceY[20], test_size = 0.3, random_state=42)
    arousalXTrain20, arousalXTest20, arousalYTrain20, arousalYTest20, valenceXTrain20, valenceXTest20, valenceYTrain20, valenceYTest20 = np.array(arousalXTrain20), np.array(arousalXTest20), np.array(arousalYTrain20), np.array(arousalYTest20), np.array(valenceXTrain20), np.array(valenceXTest20), np.array(valenceYTrain20), np.array(valenceYTest20)

    arousalXTrain21, arousalXTest21, arousalYTrain21, arousalYTest21 = train_test_split(arousalX[21], arousalY[21], test_size = 0.3, random_state=42)
    valenceXTrain21, valenceXTest21, valenceYTrain21, valenceYTest21 = train_test_split(valenceX[21], valenceY[21], test_size = 0.3, random_state=42)
    arousalXTrain21, arousalXTest21, arousalYTrain21, arousalYTest21, valenceXTrain21, valenceXTest21, valenceYTrain21, valenceYTest21 = np.array(arousalXTrain21), np.array(arousalXTest21), np.array(arousalYTrain21), np.array(arousalYTest21), np.array(valenceXTrain21), np.array(valenceXTest21), np.array(valenceYTrain21), np.array(valenceYTest21)

    arousalXTrain22, arousalXTest22, arousalYTrain22, arousalYTest22 = train_test_split(arousalX[22], arousalY[22], test_size = 0.3, random_state=42)
    valenceXTrain22, valenceXTest22, valenceYTrain22, valenceYTest22 = train_test_split(valenceX[22], valenceY[22], test_size = 0.3, random_state=42)
    arousalXTrain22, arousalXTest22, arousalYTrain22, arousalYTest22, valenceXTrain22, valenceXTest22, valenceYTrain22, valenceYTest22 = np.array(arousalXTrain22), np.array(arousalXTest22), np.array(arousalYTrain22), np.array(arousalYTest22), np.array(valenceXTrain22), np.array(valenceXTest22), np.array(valenceYTrain22), np.array(valenceYTest22)

    arousalXTrain23, arousalXTest23, arousalYTrain23, arousalYTest23 = train_test_split(arousalX[23], arousalY[23], test_size = 0.3, random_state=42)
    valenceXTrain23, valenceXTest23, valenceYTrain23, valenceYTest23 = train_test_split(valenceX[23], valenceY[23], test_size = 0.3, random_state=42)
    arousalXTrain23, arousalXTest23, arousalYTrain23, arousalYTest23, valenceXTrain23, valenceXTest23, valenceYTrain23, valenceYTest23 = np.array(arousalXTrain23), np.array(arousalXTest23), np.array(arousalYTrain23), np.array(arousalYTest23), np.array(valenceXTrain23), np.array(valenceXTest23), np.array(valenceYTrain23), np.array(valenceYTest23)

    arousalXTrain24, arousalXTest24, arousalYTrain24, arousalYTest24 = train_test_split(arousalX[24], arousalY[24], test_size = 0.3, random_state=42)
    valenceXTrain24, valenceXTest24, valenceYTrain24, valenceYTest24 = train_test_split(valenceX[24], valenceY[24], test_size = 0.3, random_state=42)
    arousalXTrain24, arousalXTest24, arousalYTrain24, arousalYTest24, valenceXTrain24, valenceXTest24, valenceYTrain24, valenceYTest24 = np.array(arousalXTrain24), np.array(arousalXTest24), np.array(arousalYTrain24), np.array(arousalYTest24), np.array(valenceXTrain24), np.array(valenceXTest24), np.array(valenceYTrain24), np.array(valenceYTest24)

    arousalXTrain25, arousalXTest25, arousalYTrain25, arousalYTest25 = train_test_split(arousalX[25], arousalY[25], test_size = 0.3, random_state=42)
    valenceXTrain25, valenceXTest25, valenceYTrain25, valenceYTest25 = train_test_split(valenceX[25], valenceY[25], test_size = 0.3, random_state=42)
    arousalXTrain25, arousalXTest25, arousalYTrain25, arousalYTest25, valenceXTrain25, valenceXTest25, valenceYTrain25, valenceYTest25 = np.array(arousalXTrain25), np.array(arousalXTest25), np.array(arousalYTrain25), np.array(arousalYTest25), np.array(valenceXTrain25), np.array(valenceXTest25), np.array(valenceYTrain25), np.array(valenceYTest25)

    arousalXTrain26, arousalXTest26, arousalYTrain26, arousalYTest26 = train_test_split(arousalX[26], arousalY[26], test_size = 0.3, random_state=42)
    valenceXTrain26, valenceXTest26, valenceYTrain26, valenceYTest26 = train_test_split(valenceX[26], valenceY[26], test_size = 0.3, random_state=42)
    arousalXTrain26, arousalXTest26, arousalYTrain26, arousalYTest26, valenceXTrain26, valenceXTest26, valenceYTrain26, valenceYTest26 = np.array(arousalXTrain26), np.array(arousalXTest26), np.array(arousalYTrain26), np.array(arousalYTest26), np.array(valenceXTrain26), np.array(valenceXTest26), np.array(valenceYTrain26), np.array(valenceYTest26)

    arousalXTrain27, arousalXTest27, arousalYTrain27, arousalYTest27 = train_test_split(arousalX[27], arousalY[27], test_size = 0.3, random_state=42)
    valenceXTrain27, valenceXTest27, valenceYTrain27, valenceYTest27 = train_test_split(valenceX[27], valenceY[27], test_size = 0.3, random_state=42)
    arousalXTrain27, arousalXTest27, arousalYTrain27, arousalYTest27, valenceXTrain27, valenceXTest27, valenceYTrain27, valenceYTest27 = np.array(arousalXTrain27), np.array(arousalXTest27), np.array(arousalYTrain27), np.array(arousalYTest27), np.array(valenceXTrain27), np.array(valenceXTest27), np.array(valenceYTrain27), np.array(valenceYTest27)

    arousalXTrain28, arousalXTest28, arousalYTrain28, arousalYTest28 = train_test_split(arousalX[28], arousalY[28], test_size = 0.3, random_state=42)
    valenceXTrain28, valenceXTest28, valenceYTrain28, valenceYTest28 = train_test_split(valenceX[28], valenceY[28], test_size = 0.3, random_state=42)
    arousalXTrain28, arousalXTest28, arousalYTrain28, arousalYTest28, valenceXTrain28, valenceXTest28, valenceYTrain28, valenceYTest28 = np.array(arousalXTrain28), np.array(arousalXTest28), np.array(arousalYTrain28), np.array(arousalYTest28), np.array(valenceXTrain28), np.array(valenceXTest28), np.array(valenceYTrain28), np.array(valenceYTest28)

    arousalXTrain29, arousalXTest29, arousalYTrain29, arousalYTest29 = train_test_split(arousalX[29], arousalY[29], test_size = 0.3, random_state=42)
    valenceXTrain29, valenceXTest29, valenceYTrain29, valenceYTest29 = train_test_split(valenceX[29], valenceY[29], test_size = 0.3, random_state=42)
    arousalXTrain29, arousalXTest29, arousalYTrain29, arousalYTest29, valenceXTrain29, valenceXTest29, valenceYTrain29, valenceYTest29 = np.array(arousalXTrain29), np.array(arousalXTest29), np.array(arousalYTrain29), np.array(arousalYTest29), np.array(valenceXTrain29), np.array(valenceXTest29), np.array(valenceYTrain29), np.array(valenceYTest29)

    arousalXTrain30, arousalXTest30, arousalYTrain30, arousalYTest30 = train_test_split(arousalX[30], arousalY[30], test_size = 0.3, random_state=42)
    valenceXTrain30, valenceXTest30, valenceYTrain30, valenceYTest30 = train_test_split(valenceX[30], valenceY[30], test_size = 0.3, random_state=42)
    arousalXTrain30, arousalXTest30, arousalYTrain30, arousalYTest30, valenceXTrain30, valenceXTest30, valenceYTrain30, valenceYTest30 = np.array(arousalXTrain30), np.array(arousalXTest30), np.array(arousalYTrain30), np.array(arousalYTest30), np.array(valenceXTrain30), np.array(valenceXTest30), np.array(valenceYTrain30), np.array(valenceYTest30)

    arousalXTrain31, arousalXTest31, arousalYTrain31, arousalYTest31 = train_test_split(arousalX[31], arousalY[31], test_size = 0.3, random_state=42)
    valenceXTrain31, valenceXTest31, valenceYTrain31, valenceYTest31 = train_test_split(valenceX[31], valenceY[31], test_size = 0.3, random_state=42)
    arousalXTrain31, arousalXTest31, arousalYTrain31, arousalYTest31, valenceXTrain31, valenceXTest31, valenceYTrain31, valenceYTest31 = np.array(arousalXTrain31), np.array(arousalXTest31), np.array(arousalYTrain31), np.array(arousalYTest31), np.array(valenceXTrain31), np.array(valenceXTest31), np.array(valenceYTrain31), np.array(valenceYTest31)

    arousalXTrain32, arousalXTest32, arousalYTrain32, arousalYTest32 = train_test_split(arousalX[32], arousalY[32], test_size = 0.3, random_state=42)
    valenceXTrain32, valenceXTest32, valenceYTrain32, valenceYTest32 = train_test_split(valenceX[32], valenceY[32], test_size = 0.3, random_state=42)
    arousalXTrain32, arousalXTest32, arousalYTrain32, arousalYTest32, valenceXTrain32, valenceXTest32, valenceYTrain32, valenceYTest32 = np.array(arousalXTrain32), np.array(arousalXTest32), np.array(arousalYTrain32), np.array(arousalYTest32), np.array(valenceXTrain32), np.array(valenceXTest32), np.array(valenceYTrain32), np.array(valenceYTest32)
   #creates/trains classifiers (since this is a DBN can now perform a regression task on continuous data)
    arousalClassifier1 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier1 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier1.fit(arousalXTrain1, arousalYTrain1)
    print('Finished Arousal Training')
    valenceClassifier1.fit(valenceXTrain1, valenceYTrain1)
    print('Finished Valence Training')

   #check accuracy, using mean squared error now that it's a regression
    arousalPredicts1 = arousalClassifier1.predict(arousalXTest1)
    valencePredicts1 = valenceClassifier1.predict(valenceXTest1)
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
    arousalClassifier2.fit(arousalXTrain2, arousalYTrain2)
    print('Finished Arousal Training')
    valenceClassifier2.fit(valenceXTrain2, valenceYTrain2)
    print('Finished Valence Training')
    arousalPredicts2 = arousalClassifier2.predict(arousalXTest2)
    valencePredicts2 = valenceClassifier2.predict(valenceXTest2)
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
    arousalClassifier3.fit(arousalXTrain3, arousalYTrain3)
    print('Finished Arousal Training')
    valenceClassifier3.fit(valenceXTrain3, valenceYTrain3)
    print('Finished Valence Training')
    arousalPredicts3 = arousalClassifier3.predict(arousalXTest3)
    valencePredicts3 = valenceClassifier3.predict(valenceXTest3)
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
    arousalClassifier4.fit(arousalXTrain4, arousalYTrain4)
    print('Finished Arousal Training')
    valenceClassifier4.fit(valenceXTrain4, valenceYTrain4)
    print('Finished Valence Training')
    arousalPredicts4 = arousalClassifier4.predict(arousalXTest4)
    valencePredicts4 = valenceClassifier4.predict(valenceXTest4)
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
    arousalClassifier5.fit(arousalXTrain5, arousalYTrain5)
    print('Finished Arousal Training')
    valenceClassifier5.fit(valenceXTrain5, valenceYTrain5)
    print('Finished Valence Training')
    arousalPredicts5 = arousalClassifier5.predict(arousalXTest5)
    valencePredicts5 = valenceClassifier5.predict(valenceXTest5)
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
    arousalClassifier6.fit(arousalXTrain6, arousalYTrain6)
    print('Finished Arousal Training')
    valenceClassifier6.fit(valenceXTrain6, valenceYTrain6)
    print('Finished Valence Training')
    arousalPredicts6 = arousalClassifier6.predict(arousalXTest6)
    valencePredicts6 = valenceClassifier6.predict(valenceXTest6)
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
    arousalClassifier7.fit(arousalXTrain7, arousalYTrain7)
    print('Finished Arousal Training')
    valenceClassifier7.fit(valenceXTrain7, valenceYTrain7)
    print('Finished Valence Training')
    arousalPredicts7 = arousalClassifier7.predict(arousalXTest7)
    valencePredicts7 = valenceClassifier7.predict(valenceXTest7)
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
    arousalClassifier8.fit(arousalXTrain8, arousalYTrain8)
    print('Finished Arousal Training')
    valenceClassifier8.fit(valenceXTrain8, valenceYTrain8)
    print('Finished Valence Training')
    arousalPredicts8 = arousalClassifier8.predict(arousalXTest8)
    valencePredicts8 = valenceClassifier8.predict(valenceXTest8)
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
    arousalClassifier9.fit(arousalXTrain9, arousalYTrain9)
    print('Finished Arousal Training')
    valenceClassifier9.fit(valenceXTrain9, valenceYTrain9)
    print('Finished Valence Training')
    arousalPredicts9 = arousalClassifier9.predict(arousalXTest9)
    valencePredicts9 = valenceClassifier9.predict(valenceXTest9)
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
    arousalClassifier10.fit(arousalXTrain10, arousalYTrain10)
    print('Finished Arousal Training')
    valenceClassifier10.fit(valenceXTrain10, valenceYTrain10)
    print('Finished Valence Training')
    arousalPredicts10 = arousalClassifier10.predict(arousalXTest10)
    valencePredicts10 = valenceClassifier10.predict(valenceXTest10)
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
    arousalClassifier11.fit(arousalXTrain11, arousalYTrain11)
    print('Finished Arousal Training')
    valenceClassifier11.fit(valenceXTrain11, valenceYTrain11)
    print('Finished Valence Training')
    arousalPredicts11 = arousalClassifier11.predict(arousalXTest11)
    valencePredicts11 = valenceClassifier11.predict(valenceXTest11)
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
    arousalClassifier12.fit(arousalXTrain12, arousalYTrain12)
    print('Finished Arousal Training')
    valenceClassifier12.fit(valenceXTrain12, valenceYTrain12)
    print('Finished Valence Training')
    arousalPredicts12 = arousalClassifier12.predict(arousalXTest12)
    valencePredicts12 = valenceClassifier12.predict(valenceXTest12)
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
    arousalClassifier13.fit(arousalXTrain13, arousalYTrain13)
    print('Finished Arousal Training')
    valenceClassifier13.fit(valenceXTrain13, valenceYTrain13)
    print('Finished Valence Training')
    arousalPredicts13 = arousalClassifier13.predict(arousalXTest13)
    valencePredicts13 = valenceClassifier13.predict(valenceXTest13)
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
    arousalClassifier14.fit(arousalXTrain14, arousalYTrain14)
    print('Finished Arousal Training')
    valenceClassifier14.fit(valenceXTrain14, valenceYTrain14)
    print('Finished Valence Training')
    arousalPredicts14 = arousalClassifier14.predict(arousalXTest14)
    valencePredicts14 = valenceClassifier14.predict(valenceXTest14)
    print('Arousal14 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest14, arousalPredicts14))+'\n')
    print('Arousal14 r^2: '+ str(metrics.r2_score(arousalYTest14, arousalPredicts14))+'\n')
    print('Valence14 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest14, valencePredicts14))+'\n')
    print('Valence14 r^2: '+ str(metrics.r2_score(valenceYTest14, valencePredicts14))+'\n')
    print('\n')

    arousalClassifier15 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier15 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier15.fit(arousalXTrain15, arousalYTrain15)
    print('Finished Arousal Training')
    valenceClassifier15.fit(valenceXTrain15, valenceYTrain15)
    print('Finished Valence Training')
    arousalPredicts15 = arousalClassifier15.predict(arousalXTest15)
    valencePredicts15 = valenceClassifier15.predict(valenceXTest15)
    print('Arousal15 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest15, arousalPredicts15))+'\n')
    print('Arousal15 r^2: '+ str(metrics.r2_score(arousalYTest15, arousalPredicts15))+'\n')
    print('Valence15 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest15, valencePredicts15))+'\n')
    print('Valence15 r^2: '+ str(metrics.r2_score(valenceYTest15, valencePredicts15))+'\n')
    print('\n')

    arousalClassifier16 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier16 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier16.fit(arousalXTrain16, arousalYTrain16)
    print('Finished Arousal Training')
    valenceClassifier16.fit(valenceXTrain16, valenceYTrain16)
    print('Finished Valence Training')
    arousalPredicts16 = arousalClassifier16.predict(arousalXTest16)
    valencePredicts16 = valenceClassifier16.predict(valenceXTest16)
    print('Arousal16 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest16, arousalPredicts16))+'\n')
    print('Arousal16 r^2: '+ str(metrics.r2_score(arousalYTest16, arousalPredicts16))+'\n')
    print('Valence16 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest16, valencePredicts16))+'\n')
    print('Valence16 r^2: '+ str(metrics.r2_score(valenceYTest16, valencePredicts16))+'\n')
    print('\n')

    arousalClassifier17 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier17 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier17.fit(arousalXTrain17, arousalYTrain17)
    print('Finished Arousal Training')
    valenceClassifier17.fit(valenceXTrain17, valenceYTrain17)
    print('Finished Valence Training')
    arousalPredicts17 = arousalClassifier17.predict(arousalXTest17)
    valencePredicts17 = valenceClassifier17.predict(valenceXTest17)
    print('Arousal17 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest17, arousalPredicts17))+'\n')
    print('Arousal17 r^2: '+ str(metrics.r2_score(arousalYTest17, arousalPredicts17))+'\n')
    print('Valence17 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest17, valencePredicts17))+'\n')
    print('Valence17 r^2: '+ str(metrics.r2_score(valenceYTest17, valencePredicts17))+'\n')
    print('\n')

    arousalClassifier18 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier18 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier18.fit(arousalXTrain18, arousalYTrain18)
    print('Finished Arousal Training')
    valenceClassifier18.fit(valenceXTrain18, valenceYTrain18)
    print('Finished Valence Training')
    arousalPredicts18 = arousalClassifier18.predict(arousalXTest18)
    valencePredicts18 = valenceClassifier18.predict(valenceXTest18)
    print('Arousal18 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest18, arousalPredicts18))+'\n')
    print('Arousal18 r^2: '+ str(metrics.r2_score(arousalYTest18, arousalPredicts18))+'\n')
    print('Valence18 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest18, valencePredicts18))+'\n')
    print('Valence18 r^2: '+ str(metrics.r2_score(valenceYTest18, valencePredicts18))+'\n')
    print('\n')

    arousalClassifier19 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier19 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier19.fit(arousalXTrain19, arousalYTrain19)
    print('Finished Arousal Training')
    valenceClassifier19.fit(valenceXTrain19, valenceYTrain19)
    print('Finished Valence Training')
    arousalPredicts19 = arousalClassifier19.predict(arousalXTest19)
    valencePredicts19 = valenceClassifier19.predict(valenceXTest19)
    print('Arousal19 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest19, arousalPredicts19))+'\n')
    print('Arousal19 r^2: '+ str(metrics.r2_score(arousalYTest19, arousalPredicts19))+'\n')
    print('Valence19 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest19, valencePredicts19))+'\n')
    print('Valence19 r^2: '+ str(metrics.r2_score(valenceYTest19, valencePredicts19))+'\n')
    print('\n')

    arousalClassifier20 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier20 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier20.fit(arousalXTrain20, arousalYTrain20)
    print('Finished Arousal Training')
    valenceClassifier20.fit(valenceXTrain20, valenceYTrain20)
    print('Finished Valence Training')
    arousalPredicts20 = arousalClassifier20.predict(arousalXTest20)
    valencePredicts20 = valenceClassifier20.predict(valenceXTest20)
    print('Arousal20 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest20, arousalPredicts20))+'\n')
    print('Arousal20 r^2: '+ str(metrics.r2_score(arousalYTest20, arousalPredicts20))+'\n')
    print('Valence20 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest20, valencePredicts20))+'\n')
    print('Valence20 r^2: '+ str(metrics.r2_score(valenceYTest20, valencePredicts20))+'\n')
    print('\n')

    arousalClassifier21 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier21 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier21.fit(arousalXTrain21, arousalYTrain21)
    print('Finished Arousal Training')
    valenceClassifier21.fit(valenceXTrain21, valenceYTrain21)
    print('Finished Valence Training')
    arousalPredicts21 = arousalClassifier21.predict(arousalXTest21)
    valencePredicts21 = valenceClassifier21.predict(valenceXTest21)
    print('Arousal21 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest21, arousalPredicts21))+'\n')
    print('Arousal21 r^2: '+ str(metrics.r2_score(arousalYTest21, arousalPredicts21))+'\n')
    print('Valence21 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest21, valencePredicts21))+'\n')
    print('Valence21 r^2: '+ str(metrics.r2_score(valenceYTest21, valencePredicts21))+'\n')
    print('\n')

    arousalClassifier22 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier22 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier22.fit(arousalXTrain22, arousalYTrain22)
    print('Finished Arousal Training')
    valenceClassifier22.fit(valenceXTrain22, valenceYTrain22)
    print('Finished Valence Training')
    arousalPredicts22 = arousalClassifier22.predict(arousalXTest22)
    valencePredicts22 = valenceClassifier22.predict(valenceXTest22)
    print('Arousal22 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest22, arousalPredicts22))+'\n')
    print('Arousal22 r^2: '+ str(metrics.r2_score(arousalYTest22, arousalPredicts22))+'\n')
    print('Valence22 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest22,valencePredicts22))+'\n')
    print('Valence22 r^2: '+ str(metrics.r2_score(valenceYTest22, valencePredicts22))+'\n')
    print('\n')

    arousalClassifier23 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier23 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier23.fit(arousalXTrain23, arousalYTrain23)
    print('Finished Arousal Training')
    valenceClassifier23.fit(valenceXTrain23, valenceYTrain23)
    print('Finished Valence Training')
    arousalPredicts23 = arousalClassifier23.predict(arousalXTest23)
    valencePredicts23 = valenceClassifier23.predict(valenceXTest23)
    print('Arousal23 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest23, arousalPredicts23))+'\n')
    print('Arousal23 r^2: '+ str(metrics.r2_score(arousalYTest23, arousalPredicts23))+'\n')
    print('Valence23 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest23, valencePredicts23))+'\n')
    print('Valence23 r^2: '+ str(metrics.r2_score(valenceYTest23, valencePredicts23))+'\n')
    print('\n')

    arousalClassifier24 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier24 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier24.fit(arousalXTrain24, arousalYTrain24)
    print('Finished Arousal Training')
    valenceClassifier24.fit(valenceXTrain24, valenceYTrain24)
    print('Finished Valence Training')
    arousalPredicts24 = arousalClassifier24.predict(arousalXTest24)
    valencePredicts24 = valenceClassifier24.predict(valenceXTest24)
    print('Arousal24 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest24, arousalPredicts24))+'\n')
    print('Arousal24 r^2: '+ str(metrics.r2_score(arousalYTest24, arousalPredicts24))+'\n')
    print('Valence24 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest24, valencePredicts24))+'\n')
    print('Valence24 r^2: '+ str(metrics.r2_score(valenceYTest24, valencePredicts24))+'\n')
    print('\n')

    arousalClassifier25 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier25 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier25.fit(arousalXTrain25, arousalYTrain25)
    print('Finished Arousal Training')
    valenceClassifier25.fit(valenceXTrain25, valenceYTrain25)
    print('Finished Valence Training')
    arousalPredicts25 = arousalClassifier25.predict(arousalXTest25)
    valencePredicts25 = valenceClassifier25.predict(valenceXTest25)
    print('Arousal25 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest25, arousalPredicts25))+'\n')
    print('Arousal25 r^2: '+ str(metrics.r2_score(arousalYTest25, arousalPredicts25))+'\n')
    print('Valence25 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest25, valencePredicts25))+'\n')
    print('Valence25 r^2: '+ str(metrics.r2_score(valenceYTest25, valencePredicts25))+'\n')
    print('\n')

    arousalClassifier26 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier26 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier26.fit(arousalXTrain26, arousalYTrain26)
    print('Finished Arousal Training')
    valenceClassifier26.fit(valenceXTrain26, valenceYTrain26)
    print('Finished Valence Training')
    arousalPredicts26 = arousalClassifier26.predict(arousalXTest26)
    valencePredicts26 = valenceClassifier26.predict(valenceXTest26)
    print('Arousal26 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest26, arousalPredicts26))+'\n')
    print('Arousal26 r^2: '+ str(metrics.r2_score(arousalYTest26, arousalPredicts26))+'\n')
    print('Valence26 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest26, valencePredicts26))+'\n')
    print('Valence26 r^2: '+ str(metrics.r2_score(valenceYTest26, valencePredicts26))+'\n')
    print('\n')

    arousalClassifier27 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier27 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier27.fit(arousalXTrain27, arousalYTrain27)
    print('Finished Arousal Training')
    valenceClassifier27.fit(valenceXTrain27, valenceYTrain27)
    print('Finished Valence Training')
    arousalPredicts27 = arousalClassifier27.predict(arousalXTest27)
    valencePredicts27 = valenceClassifier27.predict(valenceXTest27)
    print('Arousal27 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest27, arousalPredicts27))+'\n')
    print('Arousal27 r^2: '+ str(metrics.r2_score(arousalYTest27, arousalPredicts27))+'\n')
    print('Valence27 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest27, valencePredicts27))+'\n')
    print('Valence27 r^2: '+ str(metrics.r2_score(valenceYTest27, valencePredicts27))+'\n')
    print('\n')

    arousalClassifier28 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier28 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier28.fit(arousalXTrain28, arousalYTrain28)
    print('Finished Arousal Training')
    valenceClassifier28.fit(valenceXTrain28, valenceYTrain28)
    print('Finished Valence Training')
    arousalPredicts28 = arousalClassifier28.predict(arousalXTest28)
    valencePredicts28 = valenceClassifier28.predict(valenceXTest28)
    print('Arousal28 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest28, arousalPredicts28))+'\n')
    print('Arousal28 r^2: '+ str(metrics.r2_score(arousalYTest28, arousalPredicts28))+'\n')
    print('Valence28 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest28, valencePredicts28))+'\n')
    print('Valence28 r^2: '+ str(metrics.r2_score(valenceYTest28, valencePredicts28))+'\n')
    print('\n')

    arousalClassifier29 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier29 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier29.fit(arousalXTrain29, arousalYTrain29)
    print('Finished Arousal Training')
    valenceClassifier29.fit(valenceXTrain29, valenceYTrain29)
    print('Finished Valence Training')
    arousalPredicts29 = arousalClassifier29.predict(arousalXTest29)
    valencePredicts29 = valenceClassifier29.predict(valenceXTest29)
    print('Arousal29 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest29, arousalPredicts29))+'\n')
    print('Arousal29 r^2: '+ str(metrics.r2_score(arousalYTest29, arousalPredicts29))+'\n')
    print('Valence29 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest29, valencePredicts29))+'\n')
    print('Valence29 r^2: '+ str(metrics.r2_score(valenceYTest29, valencePredicts29))+'\n')
    print('\n')

    arousalClassifier30 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier30 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier30.fit(arousalXTrain30, arousalYTrain30)
    print('Finished Arousal Training')
    valenceClassifier30.fit(valenceXTrain30, valenceYTrain30)
    print('Finished Valence Training')
    arousalPredicts30 = arousalClassifier30.predict(arousalXTest30)
    valencePredicts30 = valenceClassifier30.predict(valenceXTest30)
    print('Arousal30 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest30, arousalPredicts30))+'\n')
    print('Arousal30 r^2: '+ str(metrics.r2_score(arousalYTest30, arousalPredicts30))+'\n')
    print('Valence30 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest30, valencePredicts30))+'\n')
    print('Valence30 r^2: '+ str(metrics.r2_score(valenceYTest30, valencePredicts30))+'\n')
    print('\n')

    arousalClassifier31 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier31 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier31.fit(arousalXTrain31, arousalYTrain31)
    print('Finished Arousal Training')
    valenceClassifier31.fit(valenceXTrain31, valenceYTrain31)
    print('Finished Valence Training')
    arousalPredicts31 = arousalClassifier31.predict(arousalXTest31)
    valencePredicts31 = valenceClassifier31.predict(valenceXTest31)
    print('Arousal31 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest31, arousalPredicts31))+'\n')
    print('Arousal31 r^2: '+ str(metrics.r2_score(arousalYTest31, arousalPredicts31))+'\n')
    print('Valence31 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest31, valencePredicts31))+'\n')
    print('Valence31 r^2: '+ str(metrics.r2_score(valenceYTest31, valencePredicts31))+'\n')
    print('\n')

    arousalClassifier32 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    valenceClassifier32 = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
    arousalClassifier32.fit(arousalXTrain32, arousalYTrain32)
    print('Finished Arousal Training')
    valenceClassifier32.fit(valenceXTrain32, valenceYTrain32)
    print('Finished Valence Training')
    arousalPredicts32 = arousalClassifier32.predict(arousalXTest32)
    valencePredicts32 = valenceClassifier32.predict(valenceXTest32)
    print('Arousal32 Mean Squared Error: '+str(metrics.mean_squared_error(arousalYTest32, arousalPredicts32))+'\n')
    print('Arousal32 r^2: '+ str(metrics.r2_score(arousalYTest32, arousalPredicts32))+'\n')
    print('Valence32 Mean Squared Error: '+str(metrics.mean_squared_error(valenceYTest32, valencePredicts32))+'\n')
    print('Valence32 r^2: '+ str(metrics.r2_score(valenceYTest32, valencePredicts32))+'\n')
   #end
    print('Done')

if __name__ == '__main__':
    main()