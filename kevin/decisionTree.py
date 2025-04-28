import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
import argparse

filePath = r'..\data\parkinsons.csv'


class treeNode:
    def __init__(self, dataset, attributes, originalValues, columnLabels, depth):
        self.dataset = dataset
        self.attributes = attributes
        self.originalValues = originalValues
        self.columnLabels = columnLabels
        self.depth = depth
        self.label = None
        self.children = {}
        self.isNumeric = None
        self.avg = None
    
    def addLabel(self, label, avg):
        self.label = label
        self.isNumeric = avg != None
        self.avg = avg

    def addChild(self, child, edgeLabel):
        self.children[edgeLabel] = child
    
    def getEntropy(self, data):
        labelIndex = self.columnLabels.index('label')
        datasetNP = np.array(data)
        rows, _ = np.shape(datasetNP)
        counts = {}
        for entry in datasetNP[:, labelIndex]:
            counts[entry] = counts.get(entry, 0) + 1
        total = 0
        for val in counts.values():
            total -= val/rows * math.log2(val/rows)
        return total
    
    def splitByAttribute(self, attributeIndex):
        isNumeric = self.columnLabels[attributeIndex] == 'num'
        splits = {}
        if not isNumeric:
            avg = None
            for entry in self.dataset:
                key = entry[attributeIndex]
                if key not in splits:
                    splits[key] = [entry]
                else:
                    splits[key].append(entry)
        else:
            avg = np.mean([float(row[attributeIndex]) for row in self.dataset])
            for entry in self.dataset:
                key = 'less' if float(entry[attributeIndex]) <= avg else 'greater'
                if key not in splits:
                    splits[key] = [entry]
                else:
                    splits[key].append(entry)
        return splits, avg
    
    def findBestAttribute(self, subset):
        bestAttribute = -1
        lowestVal = math.inf
        for attribute in subset:
            partitions = self.splitByAttribute(attribute)[0].values()
            partitionVal = 0
            for partition in partitions:
                partitionVal += self.getEntropy(partition) * len(partition) / len(self.dataset)
            
            if partitionVal < lowestVal:
                lowestVal = partitionVal
                bestAttribute = attribute
        return bestAttribute
    
    def build(self):
        labelIndex = self.columnLabels.index('label')

        counts = {}
        for entry in self.dataset:
            counts[entry[labelIndex]] = counts.get(entry[labelIndex], 0) + 1

        # To Do: Alter stopping criteria (max depth + minimal split)
        if len(counts) == 1 or not self.attributes:
            #or self.depth > 15 or len(self.dataset) < 3:
            self.addLabel(max(counts, key = counts.get), None)
            return self

        bestAttribute = self.findBestAttribute(self.attributes)
        newAttributes = self.attributes.copy()
        newAttributes.remove(bestAttribute)

        informationGain = self.getEntropy(self.dataset)
        partitions = list(self.splitByAttribute(bestAttribute)[0].values())
        for partition in partitions:
            informationGain -= self.getEntropy(partition) * len(partition) / len(self.dataset)

        # To Do: Alter stopping criteria (minimal gain)
        # if informationGain <= 0.000001:
        #     # Note: It's actually possible for informationGain to be < 0 due to precision errors / rounding
        #     # 0.000001
        #     # 0.01
        #     # 0
        #     self.addLabel(max(counts, key = counts.get), None)
        #     return self

        splits, avg = self.splitByAttribute(bestAttribute)
        self.addLabel(f'{bestAttribute}', avg)

        if self.isNumeric:
            for key in ['less', 'greater']:
                if key not in splits:
                    child = treeNode(self.dataset, None, self.originalValues, self.columnLabels, self.depth + 1).build()
                else:
                    child = treeNode(splits[key], newAttributes, self.originalValues, self.columnLabels, self.depth + 1).build()
                self.addChild(child, key)
        else:
            for key in self.originalValues[bestAttribute]:
                if key not in splits:
                    child = treeNode(self.dataset, None, self.originalValues, self.columnLabels, self.depth + 1).build()
                else:
                    child = treeNode(splits[key], newAttributes, self.originalValues, self.columnLabels, self.depth + 1).build()
                self.addChild(child, key)
        return self
    
    def predictClass(self, entry):
        decisionTree = self
        while decisionTree.children:
            #print(decisionTree.label)
            attributeIndex = int(decisionTree.label)
            avg = decisionTree.avg
            if not decisionTree.isNumeric:
                decisionTree = decisionTree.children[entry[attributeIndex]]
            else:
                if float(entry[attributeIndex]) <= float(avg):
                    decisionTree = decisionTree.children['less']
                else:
                    decisionTree = decisionTree.children['greater']
        return decisionTree.label
    
def getDistinctValues(dataset, columnLabels):
    rows, cols = np.shape(dataset)
    distinct = dict()
    for j in range(cols):
        if columnLabels[j] == 'cat':
            x = set()
            for i in range(rows):
                x.add(dataset[i][j])
            distinct[j] = x
    return distinct

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
