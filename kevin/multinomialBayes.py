from utils import load_training_set, load_test_set
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Helper function to calculate the probability Pr(y_i) for a two class classification instance
def classProbability(positiveInstances, negativeInstances):
    posClass = positiveInstances / (positiveInstances + negativeInstances)
    negClass = negativeInstances / (positiveInstances + negativeInstances)
    return posClass, negClass

# Given a list of 'documents' for a certain class
# We get the total frequencies of each word in the documents for that class
def getTotals(docs):
    totalFreqs = dict()
    for doc in docs:
        for word in doc:
            totalFreqs[word] = totalFreqs.get(word, 0) + 1
    return totalFreqs

# Given a word and a dictionary with the frequencies of all words for a class
# We get the probability of the word given that class
def getWordProbability(word, frequencies, vocabLen, smoothParam):
    wordFreq = frequencies.get(word, 0) + smoothParam
    total = sum(frequencies.values()) + (smoothParam * vocabLen)
    return wordFreq / total

# Given a document to be classified, and parameters to simulate the Naive Bayes model
# Output a prediction (Positive or Negative)
def classifyDoc(doc, posTotals, negTotals, posClass, negClass, vocab, useLog, smoothParam):
    uniqueWords = set(doc)
    v = len(vocab)
    p, n = (posClass, negClass) if not useLog else (np.log(posClass), np.log(negClass))
    for word in uniqueWords:
        if not useLog:
            p *= getWordProbability(word, posTotals, v, smoothParam)
            n *= getWordProbability(word, negTotals, v, smoothParam)
        else:
            p += np.log(getWordProbability(word, posTotals, v, smoothParam))
            n += np.log(getWordProbability(word, negTotals, v, smoothParam))
    # print(f"Pos: {p}, Neg: {n}")
    # If they are equal, we choose either with 50% chance (randomly)
    if p == n and random.randint(0, 1) == 0:
        return "positive"
    elif p > n:
        return "positive"
    return "negative"
   
# Given positive and negative training and test instances (and a vocab)
# Simulate the multinomial Naive Bayes algorithm
def multinomialBayes(pos_train, neg_train, vocab, pos_test, neg_test, useLog = False, smoothParam = 0):
    print(f"Using log probabilities: {useLog}")
    print(f"Using smoothing parameter: {smoothParam}")
    posClass, negClass = classProbability(len(pos_train), len(neg_train))

    posTotals = getTotals(pos_train)
    negTotals = getTotals(neg_train)

    truePos, falseNeg = 0, 0
    for entry in pos_test:
        if classifyDoc(entry, posTotals, negTotals, posClass, negClass, vocab, useLog, smoothParam) == 'positive':
            truePos += 1
        else:
            falseNeg += 1        
    
    trueNeg, falsePos = 0, 0
    for entry in neg_test:
        if classifyDoc(entry, posTotals, negTotals, posClass, negClass, vocab, useLog, smoothParam) == 'negative':
            trueNeg += 1        
        else:
            falsePos += 1
    print(f"Model accuracy: {(truePos + trueNeg) / (len(pos_test) + len(neg_test))}")
    print(f"TP: {truePos}, TN: {trueNeg}, FP: {falsePos}, FN: {falseNeg}")
    print(f"Precision: {truePos / (truePos + falsePos)}")
    print(f"Recall: {truePos / (truePos + falseNeg)}")

    # Returns accuracy
    return (truePos + trueNeg) / (len(pos_test) + len(neg_test))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", choices=[1,2,3,4,6],  required=True, type=int)
    args = parser.parse_args()

    if args.question in (1, 2):
        percentage_positive_instances_train = 0.2
        percentage_negative_instances_train = 0.2

        percentage_positive_instances_test = 0.2
        percentage_negative_instances_test = 0.2
    else:
        if args.question == 3:
            percentage_positive_instances_train = 1
            percentage_negative_instances_train = 1
        elif args.question == 4:
            percentage_positive_instances_train = 0.3
            percentage_negative_instances_train = 0.3
        else:
            percentage_positive_instances_train = 0.1
            percentage_negative_instances_train = 0.5
        percentage_positive_instances_test = 1
        percentage_negative_instances_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    if args.question == 1:
        multinomialBayes(pos_train, neg_train, vocab, pos_test, neg_test)

    elif args.question == 2:
        x = []
        y = []
        alpha = 0.0001
        while alpha <= 1000:
            x.append(alpha)
            y.append(multinomialBayes(pos_train, neg_train, vocab, pos_test, neg_test, useLog=True, smoothParam=alpha))
            alpha *= 10

        plt.semilogx(x,y)
        plt.xlabel('Value of Alpha (Smoothing Parameter)')
        plt.ylabel('Model Accuracy on Test Set') 
        plt.title('Smoothing Parameter vs. Model Accuracy')
        plt.show()

    else:
        multinomialBayes(pos_train, neg_train, vocab, pos_test, neg_test, useLog=True, smoothParam=1)
        