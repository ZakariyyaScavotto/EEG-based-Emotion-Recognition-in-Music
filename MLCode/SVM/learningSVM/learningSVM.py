#This code was written following this tutorial: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

from sklearn import datasets #library of built-in datasets
from sklearn.model_selection import train_test_split #used to split test data
from sklearn import svm #used to make svm
from sklearn import metrics #used to assess


def main():
    #loads dataset
    cancer = datasets.load_breast_cancer()

    #prints features, labels    
    print('Features: ', cancer.feature_names)
    print('Labels: ', cancer.target_names)

    #print shape of dataset
    print(cancer.data.shape)

    #print top 5 records of features (the 30 data values)
    print(cancer.data[0:5])

    #print targets
    print(cancer.target)

    #split data into trianing and testing (70% train, 30% test)
    xTrain, xTest, yTrain, yTest = train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state = 109)

    #create svm classifier
    clf = svm.SVC(kernel='linear')
    #train model
    clf.fit(xTrain,yTrain)
    #predict response for test dataset
    yPredicts = clf.predict(xTest)

    #get model accuracy
    print('Accuracy: ', metrics.accuracy_score(yTest, yPredicts))

    #model precision: result relevancy
    print("Precision:",metrics.precision_score(yTest, yPredicts))

    #model recall: how many results are truly relevant
    print("Recall:",metrics.recall_score(yTest, yPredicts))

    

if __name__ == '__main__':
    main()