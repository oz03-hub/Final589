import sys
from kNN import generatekNNGraphs
from standardBayes import NaiveBayes
import numpy as np
from kfold import stratifiedKFold
from decisionTree import treeNode, getDistinctValues
import sklearn
import randomForest as rF
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
import io

# VERY slow
def kNN(filePath):

    # kNN Algorithm
    # One-hot encoding multiple columns
    df = pd.read_csv(filePath)
    df = df.dropna()

    transformer = make_column_transformer(
        (OneHotEncoder(sparse_output=False), ['attr1_cat', 'attr4_cat', 'attr5_cat', 'attr6_cat', 'attr7_cat','attr9_cat', 'attr10_cat', 'attr11_cat', 'attr12_cat', 'attr13_cat']),
        remainder='passthrough')

    transformed = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

    csv_buffer = io.StringIO()
    transformed_df.to_csv(csv_buffer, index=False)
    generatekNNGraphs(fP=csv_buffer)

def naiveBayes(filePath):
    dataset = np.loadtxt(filePath, delimiter=",", dtype=str)
    labels = list(map(lambda x : x.split('_')[-1], dataset[0]))
    #print(labels)
    labels[labels.index('label')] = 'class'
    dataset = np.delete(dataset, 0, axis=0) # remove the header row

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
            #print(predicted)
            if predicted == entry[-1]:
                #print(predicted)
                correct += 1
        print(f"Fold {i + 1}: {correct / len(test)}")

def decisionTree(filePath):
    data = np.loadtxt(filePath, delimiter=',', dtype=str)
    _, cols = np.shape(data)
    labels = list(map(lambda x : x.split('_')[-1], data[0]))
    data = np.delete(data, 0, axis=0) # remove the header row

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
    columnLabels = list(map(lambda x : x.split('_')[-1], data[0]))
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
    filePath = r'..\data\credit_approval.csv'
    kNN(filePath)
    

