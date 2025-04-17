import matplotlib.pyplot as plt
import numpy as np
import sklearn
import argparse

filePath = r'..\data\parkinsons.csv'

def kNNAlgorithm(dataset, unknownInstance, k):
    distClassTuples = []

    # gets the euclidean distance between entry and unknown instance, excluding the classifier column
    # and stores it as a (classLabel, distance) tuple
    for entry in dataset:
        distClassTuples.append((entry[-1], np.linalg.norm(unknownInstance[:-1] - entry[:-1])))

    # sort by euclidean distance, ascending
    distClassTuples.sort(key= lambda x: x[1])

    count = 0
    for i in range(k):
        count += distClassTuples[i][0]
    
    # Return 1 if 1 is the classifier majority, else 0 (then 0 is majority)
    return 1 if count > k / 2 else 0

def mainkNN(k, normalize = True):
    # load the csv / dataset into a numpy 2d array
    data = np.loadtxt(filePath, delimiter=',', skiprows=1)
    _, cols = np.shape(data)

    # Get the maximum and minimum values for each column
    maxCols = [np.max(data[:, i]) for i in range(cols - 1)]
    minCols = [np.min(data[:, i]) for i in range(cols - 1)]
    # We do cols - 1 because we do not need to normalize the classifier column
    #print(maxCols)
    #print(minCols)
    # Normalize the dataset to the range [0, 1]
    if normalize:
        for row in data:
            for i in range(cols - 1):
                row[i] = (row[i] - minCols[i]) / (maxCols[i] - minCols[i])
    
    # Part a: shuffling the dataset
    shuffled = sklearn.utils.shuffle(data)

    # Part b: partitioning the dataset
    train, test = sklearn.model_selection.train_test_split(shuffled, test_size=0.2)

    # Running kNN algorithm on training instances
    trainingCorrect = 0
    for entry in train:
        classLabel = kNNAlgorithm(train, entry, k)
        trainingCorrect += 1 if classLabel == entry[-1] else 0
    
    trainingAccuracy = trainingCorrect / len(train)
    #print(f"Training set accuracy: {trainingAccuracy}")

    # Running kNN algorithm on testing instances
    testingCorrect = 0
    for entry in test:
        classLabel = kNNAlgorithm(train, entry, k)
        testingCorrect += 1 if classLabel == entry[-1] else 0
    
    testingAccuracy = testingCorrect / len(test)
    #print(f"Testing set accuracy: {testingAccuracy}")
    return trainingAccuracy, testingAccuracy

def generatekNNGraphs():
    print("Generating normalized graphs")
    x = []
    yTrain = []
    yTest = []
    errTrain = []
    errTest = []
    for k in range(1, 52, 2):
        print(f"k : {k}")
        training = []
        testing = []
        # for each value of k, run the algorithm 20 times
        for _ in range(20):
            tr, ts = mainkNN(k)
            training.append(tr)
            testing.append(ts)
        
        # calculate the std deviation and mean for the 20 runs
        x.append(k)
        yTrain.append(np.mean(training))
        errTrain.append(np.std(training))
        yTest.append(np.mean(testing))
        errTest.append(np.std(testing))
        print(f"Testing set accuracy: {yTest[-1]}, std: {errTest[-1]}")
        print(f"Training set accuracy: {yTrain[-1]}, std: {errTrain[-1]}")

    plt.figure(1)
    plt.errorbar(x, yTrain, yerr=errTrain, color='black', marker='.', markersize = 7, capsize=3, capthick=1)

    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy over training data)")

    plt.figure(2)
    plt.errorbar(x, yTest, yerr=errTest, color='black',marker='.', markersize = 7, capsize=3, capthick=1)

    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy over testing data)")
    plt.show()

def generateUnnormalizedkNNGraph():
    print("Generating unnormalized graph")
    x = []
    yTest = []
    errTest = []
    for k in range(1, 52, 2):
        print(f"k : {k}")
        testing = []
        # for each value of k, run the algorithm 20 times
        for _ in range(20):
            _, ts = mainkNN(k, normalize=False)
            testing.append(ts)
        
        # calculate the std deviation and mean for the 20 runs
        x.append(k)
        yTest.append(np.mean(testing))
        errTest.append(np.std(testing))
        print(f"Testing set accuracy: {yTest[-1]}, std: {errTest[-1]}")

    plt.figure(1)
    plt.errorbar(x, yTest, yerr=errTest, color='black',marker='.', markersize = 7, capsize=3, capthick=1)

    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy over testing data, unnormalized)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--unnormalized", action='store_true')

    args = parser.parse_args()
    if args.unnormalized:
        generateUnnormalizedkNNGraph()
    else:
        generatekNNGraphs()
