import pandas as pd
from kfold import StratifiedKFold
from itertools import product
from knn import KNN, normalize_data
from standard_nb import NaiveBayes
from neural_net import NeuralNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def accuracy(predictions, actual):
    c = 0
    for p, a in zip(predictions, actual):
        if p == a:
            c += 1
    return c / len(predictions)

def f1(predictions, actual):
    labels = set(actual)
    f1s = 0
    for label in labels:
        tp = 0
        fp = 0
        fn = 0

        for p, a in zip(predictions, actual):
            if p == label and a == label:
                tp += 1
            elif p == label and a != label:
                fp += 1
            elif p != label and a == label:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s += f1
    return f1s / len(labels)

def run_nn():
    digits = datasets.load_digits(return_X_y=True)
    df = pd.DataFrame(digits[0])
    df["y"] = digits[1]

    models = [
        NeuralNet(layers=[64, 100, 100, 100, 10], step_size=0.1, lambda_reg=0.1, epochs=30),
        NeuralNet(layers=[64, 100, 100, 100, 100, 100, 10], step_size=0.1, lambda_reg=0.1, epochs=30),
        NeuralNet(layers=[64, 100, 100, 100, 10], step_size=0.1, lambda_reg=0.5, epochs=30),
        NeuralNet(layers=[64, 100, 100, 100, 10], step_size=0.1, lambda_reg=0.01, epochs=30),
        NeuralNet(layers=[64, 100, 150, 100, 32, 10], step_size=0.1, lambda_reg=0.1, epochs=30),
        NeuralNet(layers=[64, 200, 10], step_size=0.1, lambda_reg=0.1, epochs=30),
    ]

    best_model = None
    best_accuracy = 0
    best_f1_score = 0

    splitter = StratifiedKFold(k=10, class_column="y")
    accuracies = []
    f1_scores = []
    t = ""
    print("NN for MNIST Dataset")
    for model in models:
        avg_accuracy = 0
        avg_f1_score = 0

        for train_split, test_split in splitter.get_splits(df):
            X_train = train_split.drop(columns=["y"]).to_numpy()
            y_train = np.eye(10)[train_split["y"].to_list()]
            X_test = test_split.drop(columns=["y"]).to_numpy()
            y_test = np.eye(10)[test_split["y"].to_list()]

            model.refit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions = [z.reshape(-1) for z in predictions]
            pred_labels = np.argmax(predictions, axis=1)
            actual_labels = np.argmax(y_test, axis=1)

            avg_accuracy += accuracy(pred_labels, actual_labels)
            avg_f1_score += f1(pred_labels, actual_labels)

        avg_accuracy /= 10
        avg_f1_score /= 10
        print(f"\tParams: {model.model_str()}")
        print(f"\tAvg Accuracy: {avg_accuracy}")
        print(f"\tAvg F1 Score: {avg_f1_score}")
        print()

        t += f"\tParams: {model.model_str()}\n\tAvg Accuracy: {avg_accuracy}\n\tAvg F1 Score: {avg_f1_score}\n"

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_f1_score = avg_f1_score
            best_model = model

        accuracies.append(avg_accuracy)
        f1_scores.append(avg_f1_score)

    print(f"Best Model: {best_model.model_str()}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best F1 Score: {best_f1_score}")

    with open("ozel/results/mnist_nn.txt", "w") as f:
        f.write(t)
        f.write("Best metrics\n")
        f.write(f"Best Model: {best_model.model_str()}\n")
        f.write(f"Best Accuracy: {best_accuracy}\n")
        f.write(f"Best F1 Score: {best_f1_score}\n")

if __name__ == "__main__":
    run_nn()
