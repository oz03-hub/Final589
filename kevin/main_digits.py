from sklearn import datasets 
import numpy as np
import matplotlib.pyplot as plt
import sys
from kNN import generatekNNGraphs
from standardBayes import NaiveBayes
import numpy as np
from kfold import stratifiedKFold
from decisionTree import treeNode, getDistinctValues
import sklearn
import randomForest as rF
import pandas as pd
import io

def kNN(filePath):
    # kNN Algorithm
    generatekNNGraphs(fP=filePath)

def naiveBayes(filePath):
    dataset = np.loadtxt(filePath, delimiter=",", skiprows=1)
    labels = []
    for i in range(len(dataset[0]) - 1):
        labels.append('num')
    labels.append('class')

    folds = stratifiedKFold(dataset, LabelIndex=-1, k=10)

    for i in range(len(folds)):
        train = []
        test = []
        for j in range(len(folds)):
            if i != j:
                train += folds[j]
            else:
                test += folds[j]
        model = NaiveBayes(train, labels=labels, smoothParam=1)
        correct = 0
        for entry in test:
            predicted = model.fit(entry)
            print(predicted)
            if predicted == entry[-1]:
                #print(predicted)
                correct += 1
        print(f"Fold {i + 1}: {correct / len(test)}")

def decisionTree(filePath):
    data = np.loadtxt(filePath, delimiter=',', skiprows=1, dtype=str)
    _, cols = np.shape(data)
    labels = []
    for i in range(len(data[0]) - 1):
        labels.append('num')
    labels.append('label')
    distinctVals = getDistinctValues(data, labels)

    testingAccuracy, trainingAccuracy = [], []
    for _ in range(100):
        # Part a: shuffling the dataset
        shuffled = sklearn.utils.shuffle(data)

        # Part b: partitioning the dataset
        train, test = sklearn.model_selection.train_test_split(shuffled, test_size=0.2)

        dT = treeNode(train, [i for i, label in enumerate(labels) if label != 'label'], distinctVals, labels, 0).build()

        correctTest = 0
        for entry in test:
            #print(dT.predictClass(entry))
            correctTest += 1 if dT.predictClass(entry) == entry[-1] else 0
        testingAccuracy.append(correctTest / len(test))

        correctTrain = 0
        for entry in train:
            correctTrain += 1 if dT.predictClass(entry) == entry[-1] else 0
        trainingAccuracy.append(correctTrain / len(train))

    print(f"Testing accuracies: {testingAccuracy}\nMean:{np.mean(testingAccuracy)}\nStd:{np.std(testingAccuracy)}")
    print(f"Training accuracies: {trainingAccuracy}\nMean:{np.mean(trainingAccuracy)}\nStd:{np.std(trainingAccuracy)}")

def randomForest(filePath, numTrees=30, numFolds=10):
    data = np.loadtxt(filePath, delimiter=',', dtype=str)
    columnLabels = []
    for i in range(len(data[0]) - 1):
        columnLabels.append('num')
    columnLabels.append('label')
    labelIndex = columnLabels.index('label')

    attributes = [i for i, label in enumerate(columnLabels) if label != 'label']

    data = np.delete(data, 0, axis=0) # remove the header row

    stratified_kfolds = stratifiedKFold(data, labelIndex, numFolds)
    
    distinctVals = getDistinctValues(data, columnLabels)

    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    for i in range(numFolds):
        test = stratified_kfolds[i]
        training = np.concatenate([x for j, x in enumerate(stratified_kfolds) if j != i])

        randomForest = rF.generateRandomForest(training, numTrees, attributes, distinctVals, columnLabels)
        TP, FP, TN, FN = 0,0,0,0
        correct = 0
        #To Do: Update this to accomodate for multiple classes
        for entry in test:
            result = rF.majorityVote(randomForest, entry)
            if result == entry[labelIndex]:
                correct += 1
            # if result == entry[labelIndex] and entry[labelIndex] == '1':
            #     TP += 1
            # elif result == entry[labelIndex] and entry[labelIndex] == '0':
            #     TN += 1
            # elif result != entry[labelIndex] and entry[labelIndex] == '0':
            #     FN += 1
            # elif result != entry[labelIndex] and entry[labelIndex] == '1':
            #     FP += 1
        #print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        accuracies.append(correct / len(test)) 
    print(accuracies)
    print(np.average(accuracies))
    return np.average(accuracies)
    # Random Forest Algorithm
    

if __name__ == "__main__":
    digits = datasets.load_digits(return_X_y=True) 
    digits_dataset_X = pd.DataFrame(digits[0])
    digits_dataset_y = pd.DataFrame(digits[1])
    digits_dataset_X.columns = [str(column) + '_num' for column in digits_dataset_X.columns]
    digits_dataset_y = digits_dataset_y.rename(columns={0: 'label'})
    
    df = pd.concat([pd.DataFrame(digits_dataset_X), pd.DataFrame(digits_dataset_y)], axis=1)
    filePath = io.StringIO()
    df.to_csv(filePath, index=False)
    filePath.seek(0) 
    #kNN(filePath)
    #naiveBayes(filePath)
    #decisionTree(filePath)
    randomForest(filePath, numTrees=30, numFolds=10)

    

