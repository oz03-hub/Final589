import matplotlib.pyplot as plt
import numpy as np
import math

filePath = r'dataset\wdbc.csv'
# To Do: Alter filePath based on dataset you want to test

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
        self.isNumeric = self.columnLabels[int(label)] == 'num'
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

        subset = np.random.choice(self.attributes, int(math.sqrt(len(self.attributes))), replace=False)
        bestAttribute = self.findBestAttribute(subset)

        informationGain = self.getEntropy(self.dataset)
        partitions = list(self.splitByAttribute(bestAttribute)[0].values())
        for partition in partitions:
            informationGain -= self.getEntropy(partition) * len(partition) / len(self.dataset)

        # To Do: Alter stopping criteria (minimal gain)
        if informationGain <= 0.01:
            # Note: It's actually possible for informationGain to be < 0 due to precision errors / rounding
            # 0.000001
            # 0.01
            # 0
            self.addLabel(max(counts, key = counts.get), None)
            return self

        splits, avg = self.splitByAttribute(bestAttribute)
        self.addLabel(f'{bestAttribute}', avg)

        if self.isNumeric:
            for key in ['less', 'greater']:
                if key not in splits:
                    child = treeNode(self.dataset, None, self.originalValues, self.columnLabels, self.depth + 1).build()
                else:
                    child = treeNode(splits[key], self.attributes, self.originalValues, self.columnLabels, self.depth + 1).build()
                self.addChild(child, key)
        else:
            for key in self.originalValues[bestAttribute]:
                if key not in splits:
                    child = treeNode(self.dataset, None, self.originalValues, self.columnLabels, self.depth + 1).build()
                else:
                    child = treeNode(splits[key], self.attributes, self.originalValues, self.columnLabels, self.depth + 1).build()
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
    
def stratifiedKFold(dataset, LabelIndex, k = 5):
    splitByClass = dict()
    for entry in dataset:
        label = entry[LabelIndex]
        if label not in splitByClass:
            splitByClass[label] = []
        splitByClass[label].append(entry)
    #print(splitByClass)
    folds = [[] for _ in range(k)]

    for _, entries in splitByClass.items():
        foldSize = len(entries) // k
        # Here we round down to ensure that all folds have the same number of instances (but this means we may possibly exclude <= 4 instances per class)
        for i in range(k):
            for j in range(foldSize):
                if not entries:
                    break
                randomIndex = np.random.randint(0, len(entries))
                folds[i].append(entries[randomIndex])
                entries.pop(randomIndex)   
    #print(folds)
    return folds

def majorityVote(decisionTrees, entry):
    votes = {}
    for tree in decisionTrees:
        prediction = tree.predictClass(entry)
        votes[prediction] = votes.get(prediction, 0) + 1
    #print(votes)
    return max(votes, key=votes.get)

def generateBootstrapSamples(dataset, sampleSize):
    rows, _ = np.shape(dataset)
    output = []
    for _ in range(sampleSize):
        randomIndex = np.random.randint(0, rows)
        output.append(dataset[randomIndex])
    return np.array(output)

def generateRandomForest(dataset, numTrees, attributes, originalValues, columnLabels):
    forest = []
    for _ in range(numTrees):
        sample = generateBootstrapSamples(dataset, len(dataset))
        tree = treeNode(sample, attributes, originalValues, columnLabels, 1).build()
        forest.append(tree)
    return forest

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

def main(numTrees = 1, numFolds = 5):
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

        randomForest = generateRandomForest(training, numTrees, attributes, distinctVals, columnLabels)
        TP, FP, TN, FN = 0,0,0,0
        for entry in test:
            result = majorityVote(randomForest, entry)
            if result == entry[labelIndex] and entry[labelIndex] == '1':
                TP += 1
            elif result == entry[labelIndex] and entry[labelIndex] == '0':
                TN += 1
            elif result != entry[labelIndex] and entry[labelIndex] == '1':
                FN += 1
            elif result != entry[labelIndex] and entry[labelIndex] == '0':
                FP += 1
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

def generateGraphs():
    numTrees = [1, 5, 10, 20, 30, 40, 50]
    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    for i in numTrees:
        accuracy, precision, recall, f_score = main(numTrees=i)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)

    plt.figure(1)
    plt.plot(numTrees, accuracies, label='Accuracy')
    plt.title("Random Forest Accuracy vs Number of Trees")
    plt.xlabel("Number of Trees")    
    plt.ylabel("Accuracy")

    plt.figure(2)
    plt.title("Random Forest Precision vs Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Precision")
    plt.plot(numTrees, precisions, label='Precision')

    plt.figure(3)
    plt.title("Random Forest Recall vs Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Recall")
    plt.plot(numTrees, recalls, label='Recall')

    plt.figure(4)
    plt.title("Random Forest F-Score vs Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("F-Score")
    plt.plot(numTrees, f_scores, label='F-Score')
    plt.show()

if __name__ == "__main__":
    generateGraphs()
