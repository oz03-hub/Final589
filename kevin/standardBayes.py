import numpy as np
import math

class NaiveBayes:
    def __init__(self, dataset, labels, smoothParam=0):
        self.dataset = dataset
        self.labels = labels
        self.smoothParam = smoothParam

    def splitByClass(self):
        classIndex = self.labels.index("class")
        classes = dict()
        for instance in self.dataset:
            classLabel = instance[classIndex]
            if classLabel not in classes:
                classes[classLabel] = []
            classes[classLabel].append(instance)
        return classes
    
    def getCategoricalAttributeProbabilities(self, instances, attributeIndex):
        attributeProb = dict()
        for instance in instances:
            attributeValue = instance[attributeIndex]
            if attributeValue not in attributeProb:
                attributeProb[attributeValue] = 0
            attributeProb[attributeValue] += 1
        return attributeProb
    
    def getNumericalAttributeProbabilities(self, instances, attributeIndex):
        numbers = []
        for instance in instances:
            numbers.append(float(instance[attributeIndex]))
        mean = np.mean(numbers)
        std = np.std(numbers)
        #print(mean)
        return mean, std
    
    def fit(self, instance):
        classProb = self.splitByClass()
        maxClass = None
        maxProb = -math.inf
        for classLabel, instances in classProb.items():
            prob = np.log(len(instances) / len(self.dataset))
            for i, label in enumerate(self.labels):
                if label == 'cat':
                    attributeProb = self.getCategoricalAttributeProbabilities(instances, i)
                    prob += np.log((attributeProb.get(instance[i], 0) + self.smoothParam) / (len(instances) + self.smoothParam * len(attributeProb)))
                elif label == 'num':
                    mean, std = self.getNumericalAttributeProbabilities(instances, i)
                    if std == 0:
                        guassian = 0 if float(instance[i]) != mean else 1
                    else:
                        guassian = 1 / (math.sqrt(2 * math.pi * std ** 2)) * np.exp(-((float(instance[i]) - mean) ** 2) / (2 * std ** 2))
                    if guassian == 0:
                        prob += np.log(1e-10)
                    else:
                        prob += np.log(guassian)
            #print(f"Class: {classLabel}, Probability: {prob}")
            if prob > maxProb:
                maxProb = prob
                maxClass = classLabel
        return maxClass
    

        