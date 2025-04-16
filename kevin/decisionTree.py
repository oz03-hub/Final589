import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
import argparse

filePath = r'datasets\car.csv'

# Class used to build the decision tree
# Two functions ->
# addLabel to add a label to a node
# addChild to add an edge from node to child (with edge label)
class treeNode:
    def __init__(self):
        self.label = None
        self.children = {}
    
    def addLabel(self, label):
        self.label = label

    def addChild(self, child, edgeLabel):
        self.children[edgeLabel] = child

# Function that, given a dataset (numpy 2d array)
# Returns the distinct values of each column as a list of sets
def getDistinctValues(dataset):
    rows, cols = np.shape(dataset)
    distinct = []
    # cols - 1 because we don't need the class label column
    for j in range(cols - 1):
        x = set()
        for i in range(rows):
            x.add(dataset[i][j])
        distinct.append(x)
    # iterate through every column, get all possible distinct values of each col
    return distinct

# Function that, given a dataset (2d array, numpy or not)
# Returns the calculated entropy based on the last column (class label column)
def getEntropy(dataset):
    datasetNP = np.array(dataset)
    rows, _ = np.shape(datasetNP)
    counts = {}
    for entry in datasetNP[:,-1]:
        counts[entry] = counts.get(entry, 0) + 1
    total = 0
    for val in counts.values():
        total -= val/rows * math.log2(val/rows)
    return total

def getGini(dataset):
    datasetNP = np.array(dataset)
    rows, _ = np.shape(datasetNP)
    counts = {}
    for entry in datasetNP[:,-1]:
        counts[entry] = counts.get(entry, 0) + 1
    total = 1
    for val in counts.values():
        total -= (val / rows) ** 2
    return total

# Function that, given a dataset (2d numpy array) and attributeIndex (column index to split on)
# Returns a partitioned dataset as a dict with key -> attributeValue, value -> partition
def splitByAttribute(dataset, attributeIndex):
    splits = {}
    for entry in dataset:
        key = entry[attributeIndex]
        if key not in splits:
            splits[key] = [entry]
        else:
            splits[key].append(entry)
    return splits

# Function that, given a dataset (2d numpy array) and a list of possible attributes (column indices)
# Returns the best attribute to split on from that input list
def findBestAttribute(dataset, attributes, isGini):
    bestAttribute = -1
    lowestVal = math.inf
    for attribute in attributes:
        # For each attribute, partition the dataset based on that attribute's values
        partitions = splitByAttribute(dataset, attribute).values()

        partitionVal = 0
        for partition in partitions:
            partitionVal += (getGini(partition) if isGini else getEntropy(partition)) * len(partition) / len(dataset)
        # Calculate the weighted average entropy / gini of the resulting partitions

        if partitionVal < lowestVal:
            lowestVal = partitionVal
            bestAttribute = attribute
    # Return the attribute which results in the lowest partition entropy / gini
    return bestAttribute


# Function that, given a dataset(2d numpy array), a list of possible attributes (column indices)
# and the set of original distinct values for each attribute / column
# -> Recursively builds a decision tree
def decision_tree(dataset, attributes, originalValues, gini, prune):
    newNode = treeNode()

    # Gets the counts for each class
    counts = {}
    for entry in dataset:
        counts[entry[-1]] = counts.get(entry[-1], 0) + 1
    
    # Stopping criteria
    # len(counts) == 1 <=> all instances belonging to same class
    # not attributes <=> no more attributes to test
    # max(count.values()) > 0.85 <=> when more than 85% of instances are of same class
    # In both cases, we label this node with the majority class
    if len(counts) == 1 or not attributes or (prune and max(counts.values()) / len(dataset)  > 0.85):
        newNode.addLabel(max(counts, key = counts.get))
        return newNode

    bestAttribute = findBestAttribute(dataset, attributes, gini)
    # Defines newNode as a decision node that tests bestAttribute
    newNode.addLabel(bestAttribute)
    # We make a copy of the attributes array to avoid shared references
    newAttributes = attributes.copy()
    newAttributes.remove(bestAttribute)

    splits = splitByAttribute(dataset, bestAttribute)
    # originalValues[bestAttribute] <=> list of all different values of bestAttribute
    for key in originalValues[bestAttribute]:
        # key not in splits <=> D_v is empty
        if key not in splits:
            # Calling decision_tree with attribute = None results in a leaf node labelled with majority class of dataset
            child = decision_tree(dataset, None, originalValues, gini, prune)
        else:
            # Creates a subtree with this partition and 1 less attribute
            child = decision_tree(splits[key], newAttributes, originalValues, gini, prune)
        # an edge from newNode to child with label = attributeValue
        newNode.addChild(child, key)
    return newNode

# Function that, given a decision tree and an entry (row of data)
# Returns a predicted class
def predictClass(decisionTree, entry):
    # While we're at a decision node / not a leaf node
    while decisionTree.children:
        # Go to the child node corresponding to the attribute value of our entry
        attributeIndex = decisionTree.label
        decisionTree = decisionTree.children[entry[attributeIndex]]
    return decisionTree.label
        

def main(gini, prune):
    # load the csv / dataset into a numpy 2d array
    data = np.loadtxt(filePath, delimiter=',', skiprows=1, dtype=str)
    _, cols = np.shape(data)
    distinctVals = getDistinctValues(data)

    testingAccuracy, trainingAccuracy = [], []
    for _ in range(100):
        # Part a: shuffling the dataset
        shuffled = sklearn.utils.shuffle(data)

        # Part b: partitioning the dataset
        train, test = sklearn.model_selection.train_test_split(shuffled, test_size=0.2)

        dT = decision_tree(train, list(range(cols - 1)), distinctVals, gini, prune)

        correctTest = 0
        for entry in test:
            correctTest += 1 if predictClass(dT, entry) == entry[-1] else 0
        testingAccuracy.append(correctTest / len(test))

        correctTrain = 0
        for entry in train:
            correctTrain += 1 if predictClass(dT, entry) == entry[-1] else 0
        trainingAccuracy.append(correctTrain / len(train))

    print(f"Testing accuracies: {testingAccuracy}\nMean:{np.mean(testingAccuracy)}\nStd:{np.std(testingAccuracy)}")
    print(f"Training accuracies: {trainingAccuracy}\nMean:{np.mean(trainingAccuracy)}\nStd:{np.std(trainingAccuracy)}")

    plt.figure(1)
    plt.hist(testingAccuracy,bins=50, range=[0.85, 1.0])

    plt.xlabel("(Accuracy)")    
    plt.ylabel("(Accuracy Frequency on Testing Data)")

    plt.figure(2)
    plt.hist(trainingAccuracy, bins=50, range=[0.9,1.0])

    plt.xlabel("(Accuracy)")
    plt.ylabel("(Accuracy Frequency on Training Data)")
    plt.show()
    

    return testingAccuracy, trainingAccuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gini", action='store_true')
    parser.add_argument("-p", "--prune", action='store_true')
    args = parser.parse_args()
    main(args.gini, args.prune)
