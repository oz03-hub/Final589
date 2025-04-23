import sys
from kNN import generatekNNGraphs
from standardBayes import NaiveBayes
import numpy as np
from kfold import stratifiedKFold
from decisionTree import treeNode, getDistinctValues
import sklearn
import randomForest as rF
import matplotlib.pyplot as plt

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

    x = [0.1, 0.5, 1, 2, 5, 10]
    accuracy = []
    fscore = []
    for smooth in x:
        foldAccuracies = []
        foldFScores = []
        for i in range(len(folds)):
            train = []
            test = []
            for j in range(len(folds)):
                if i != j:
                    train += folds[j]
                else:
                    test += folds[j]
            model = NaiveBayes(train, labels=labels, smoothParam=smooth)
            correct = 0
            instances = dict()
            for entry in test:
                classLabel = model.fit(entry)
                if classLabel == entry[-1]:
                    correct += 1
                if classLabel not in instances:
                    instances[classLabel] = [0,0,0] # TP, FN, FP for each class
                instances[classLabel][0] += 1 # TP
            else:
                if entry[-1] not in instances:
                    instances[entry[-1]] = [0,0,0] # TP, FN, FP for each class
                instances[entry[-1]][1] += 1 # FN
                if classLabel not in instances:
                    instances[classLabel] = [0,0,0] # TP, FN, FP for each class
                instances[classLabel][2] += 1 # FP
            foldAccuracies.append(correct / len(test))
            f_score = 0
            for key in instances.keys():
                precision = instances[key][0] / (instances[key][0] + instances[key][2]) if (instances[key][0] + instances[key][2]) != 0 else 0
                recall = instances[key][0] / (instances[key][0] + instances[key][1]) if (instances[key][0] + instances[key][1]) != 0 else 0
                f_score += (2 * ((precision * recall) / (precision + recall))) if (precision + recall) != 0 else 0
            f_score /= len(instances.keys())
            foldFScores.append(f_score)
        print(f"Alpha: {smooth}, Accuracy: {np.mean(foldAccuracies)}, F-Score: {np.mean(foldFScores)}")
        accuracy.append(np.mean(foldAccuracies))
        fscore.append(np.mean(foldFScores))
    
    plt.figure(1)
    plt.plot(x, accuracy, label='Accuracy', marker='o')
    plt.plot(x, fscore, label='F-Score', marker='o')
    plt.xlabel("(Value of Smoothing Parameter)")
    plt.ylabel("(Accuracy and F-Score over training data)")
    plt.legend()
    plt.grid(True)
    plt.title("Standard Naive Bayes")
    plt.show()
    

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

        #To Do: Update this to accomodate for multiple classes
        for entry in test:
            result = rF.majorityVote(randomForest, entry)
            if result == entry[labelIndex] and entry[labelIndex] == '1':
                TP += 1
            elif result == entry[labelIndex] and entry[labelIndex] == '0':
                TN += 1
            elif result != entry[labelIndex] and entry[labelIndex] == '0':
                FN += 1
            elif result != entry[labelIndex] and entry[labelIndex] == '1':
                FP += 1
        #print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        accuracies.append((TP + TN) / (TP + FP + TN + FN))
        precisions.append(TP / (TP + FP))
        recalls.append(TP / (TP + FN))
        f_scores.append(2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))    
    print(accuracies)
    print(np.average(accuracies))
    print(precisions)
    print(np.average(precisions))
    print(recalls)
    print(np.average(recalls))
    print(f_scores)
    print(np.average(f_scores))
    return np.average(accuracies), np.average(precisions), np.average(recalls), np.average(f_scores)
    # Random Forest Algorithm
    

if __name__ == "__main__":
    import os
    filePath = os.path.join(os.path.dirname(__file__), '..', 'data', 'parkinsons.csv')
    naiveBayes(filePath)
    

