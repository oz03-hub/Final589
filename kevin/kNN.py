import matplotlib.pyplot as plt
import numpy as np
import sklearn
import argparse
import io
import kfold

filePath = r'..\data\parkinsons.csv'

def kNNAlgorithm(dataset, unknownInstance, k):
    distClassTuples = []

    # gets the euclidean distance between entry and unknown instance, excluding the classifier column
    # and stores it as a (classLabel, distance) tuple
    for entry in dataset:
        distClassTuples.append((entry[-1], np.linalg.norm(unknownInstance[:-1] - entry[:-1])))

    # sort by euclidean distance, ascending
    distClassTuples.sort(key= lambda x: x[1])
    #print(distClassTuples[:5])
    #print(k)
    count = dict()
    for i in range(k):
        count[distClassTuples[i][0]] = count.get(distClassTuples[i][0], 0) + 1
    
    #print(count)
    # Return 1 if 1 is the classifier majority, else 0 (then 0 is majority)
    return max(count, key=count.get)

def mainkNN(k, folds ):
    #print(k)
    # load the csv / dataset into a numpy 2d array
    #print(data[0])

    accuracies = []
    fscores = []
    for i in range(len(folds)):
        train = []
        test = []
        for j in range(len(folds)):
            if i != j:
                train += folds[j]
            else:
                test = folds[j]
        instances = dict()
        correct = 0
        for entry in test:
            classLabel = kNNAlgorithm(train, entry, k)
            #print(classLabel, entry[-1])
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
        accuracies.append(correct / len(test))
        f_score = 0
        for key in instances.keys():
            precision = instances[key][0] / (instances[key][0] + instances[key][2]) if (instances[key][0] + instances[key][2]) != 0 else 0
            recall = instances[key][0] / (instances[key][0] + instances[key][1]) if (instances[key][0] + instances[key][1]) != 0 else 0
            f_score += (2 * ((precision * recall) / (precision + recall))) if (precision + recall) != 0 else 0
        f_score /= len(instances.keys())
        fscores.append(f_score)

        #print(instances)
    return accuracies, fscores

def generatekNNGraphs(fP=filePath):
    print("Generating normalized graphs")
    x = []
    accuracy = []
    fscore = []
    if isinstance(fP, io.StringIO):
        fP.seek(0)
    data = np.loadtxt(fP, delimiter=',', skiprows=1, dtype=str)
    _, cols = np.shape(data)
    data = data.astype(object)
    for i in range(cols - 1):
        data[:, i] = data[:, i].astype(float)
    #print(data[0])

    # Get the maximum and minimum values for each column
    maxCols = [np.max(data[:, i]) for i in range(cols - 1)]
    minCols = [np.min(data[:, i]) for i in range(cols - 1)]

    for row in data:
            for i in range(cols - 1):
                row[i] = (row[i] - minCols[i]) / (maxCols[i] - minCols[i]) if maxCols[i] - minCols[i] != 0 else 0
    
    # Part a: generating k folds
    folds = kfold.stratifiedKFold(data, LabelIndex=-1, k=10)

    for k in [1, 3, 5, 10, 20, 30]:
        print(f"k : {k}")
        acc, f = mainkNN(k = k, folds = folds)
        
        x.append(k)
        accuracy.append(np.mean(acc))
        fscore.append(np.mean(f))
        print(f"Testing set accuracy: {accuracy[-1]}, fscore: {fscore[-1]}")

    plt.figure(1)
    plt.plot(x, accuracy, label='Accuracy', marker='o')
    plt.plot(x, fscore, label='F-Score', marker='o')
    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy and F-Score over training data)")
    plt.legend()
    plt.grid(True)
    plt.title("Normalized kNN")
    plt.show()
