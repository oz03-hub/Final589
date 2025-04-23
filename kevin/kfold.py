import numpy as np

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