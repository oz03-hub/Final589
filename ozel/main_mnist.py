import pandas as pd
from kfold import StratifiedKFold
from itertools import product
from knn import KNN, normalize_data
from random_forest import RandomForest
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

def run_knn():
    digits = datasets.load_digits(return_X_y=True)
    df = pd.DataFrame(digits[0])
    df["y"] = digits[1]
    df.columns = df.columns.astype(str)
    df = normalize_data(df)

    splitter = StratifiedKFold(k=10, class_column="y")

    hyperparams = {
        "k": [1, 3, 5, 10, 20, 30],
    }

    param_grid = product(*hyperparams.values())
    param_names = list(hyperparams.keys())

    best_params = None
    best_accuracy = 0
    best_f1_score = 0

    accuracies = []
    f1_scores = []
    t = ""
    print("KNN for Parkinson's Dataset")
    for params in param_grid:
        params_dict = dict(zip(param_names, params))
        
        avg_accuracy = 0
        avg_f1_score = 0
        for train_split, test_split in splitter.get_splits(df):
            X_train = train_split.drop(columns=["y"])
            y_train = train_split["y"]
            X_test = test_split.drop(columns=["y"])
            y_test = test_split["y"]

            knn_model = KNN(k=params_dict["k"])
            knn_model.fit(X_train, y_train)

            predictions = knn_model.predict(X_test)
            actual = y_test.to_list()
            avg_accuracy += accuracy(predictions, actual)
            avg_f1_score += f1(predictions, actual)
        
        avg_accuracy /= 10
        avg_f1_score /= 10
        print(f"\tParams: {params_dict}")
        print(f"\tAvg Accuracy: {avg_accuracy}")
        print(f"\tAvg F1 Score: {avg_f1_score}")
        print()

        t += f"\tParams: {params_dict}\n\tAvg Accuracy: {avg_accuracy}\n\tAvg F1 Score: {avg_f1_score}\n"

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_f1_score = avg_f1_score
            best_params = params_dict

        accuracies.append(avg_accuracy)
        f1_scores.append(avg_f1_score)

    print(f"Best Params: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best F1 Score: {best_f1_score}")

    with open("ozel/results/mnist_knn.txt", "w") as f:
        f.write(t)
        f.write("Best metrics\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Accuracy: {best_accuracy}\n")
        f.write(f"Best F1 Score: {best_f1_score}\n")
    
    plt.plot(hyperparams["k"], accuracies, label="Accuracy", marker="o")
    plt.plot(hyperparams["k"], f1_scores, label="F1 Score", marker="o")
    plt.ylim(min(*accuracies, *f1_scores) - 0.05, max(*accuracies, *f1_scores) + 0.05)
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy and F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig("ozel/figures/mnist_knn.png")
    plt.clf()

def run_rf():
    digits = datasets.load_digits(return_X_y=True)
    df = pd.DataFrame(digits[0])
    df["y"] = digits[1]
    df.columns = df.columns.astype(str)

    splitter = StratifiedKFold(k=10, class_column="y")

    hyperparams = {
        "n_tree": [2, 5, 10, 20, 30, 50]
    }

    param_grid = product(*hyperparams.values())
    param_names = list(hyperparams.keys())
    
    best_params = None
    best_accuracy = 0
    best_f1_score = 0

    print("Random Forest for Parkinson's Dataset")
    accuracies = []
    f1_scores = []
    t = ""
    for params in param_grid:
        params_dict = dict(zip(param_names, params))

        avg_accuracy = 0
        avg_f1_score = 0
        for train_split, test_split in splitter.get_splits(df):
            rf_model = RandomForest(n_tree=params_dict["n_tree"], data=train_split, numeric_end="", class_column="y", stopping_criterion_hyperparameter=0.1)

            predictions = rf_model.predict_df(test_split)
            actual = test_split["y"].to_list()
            avg_accuracy += accuracy(predictions, actual)
            avg_f1_score += f1(predictions, actual)
        
        avg_accuracy /= 10
        avg_f1_score /= 10
        print(f"\tParams: {params_dict}")
        print(f"\tAvg Accuracy: {avg_accuracy}")
        print(f"\tAvg F1 Score: {avg_f1_score}")
        print()

        t += f"\tParams: {params_dict}\n\tAvg Accuracy: {avg_accuracy}\n\tAvg F1 Score: {avg_f1_score}\n"

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_f1_score = avg_f1_score
            best_params = params_dict

        accuracies.append(avg_accuracy)
        f1_scores.append(avg_f1_score)

    print(f"Best Params: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best F1 Score: {best_f1_score}")

    with open("ozel/results/mnist_rf.txt", "w") as f:
        f.write(t)
        f.write("Best metrics\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Accuracy: {best_accuracy}\n")
        f.write(f"Best F1 Score: {best_f1_score}\n")
    
    plt.plot(hyperparams["n_tree"], accuracies, label="Accuracy", marker="o")
    plt.plot(hyperparams["n_tree"], f1_scores, label="F1 Score", marker="o")
    plt.ylim(min(*accuracies, *f1_scores) - 0.05, max(*accuracies, *f1_scores) + 0.05)
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy and F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig("ozel/figures/mnist_rf.png")
    plt.clf()


def run_nn():
    digits = datasets.load_digits(return_X_y=True)
    df = pd.DataFrame(digits[0])
    df["y"] = digits[1]

    models = [
        NeuralNet(layers=[64, 48, 10], step_size=0.1, lambda_reg=0.1, batch_size=50, epochs=50),
        NeuralNet(layers=[64, 100, 10], step_size=0.1, lambda_reg=0.1, batch_size=50, epochs=50),
        NeuralNet(layers=[64, 200, 10], step_size=0.1, lambda_reg=0.1, epochs=50),
        NeuralNet(layers=[64, 100, 100, 10], step_size=0.1, lambda_reg=0.1, batch_size=50, epochs=50),
        NeuralNet(layers=[64, 100, 100, 100, 100, 100, 10], step_size=0.1, lambda_reg=0.1, epochs=50),
        NeuralNet(layers=[64, 100, 100, 10], step_size=0.1, lambda_reg=0.5, batch_size=50, epochs=50),
        NeuralNet(layers=[64, 100, 100, 10], step_size=0.1, lambda_reg=0.9, batch_size=50, epochs=50),
    ]

    best_model = None
    best_accuracy = 0
    best_f1_score = 0

    splitter = StratifiedKFold(k=10, class_column="y")
    accuracies = []
    f1_scores = []
    best_losses = None
    t = ""
    print("NN for MNIST Dataset")
    for model in models:
        avg_accuracy = 0
        avg_f1_score = 0
        
        losses = [0 for _ in range(model.epochs)]
        for train_split, test_split in splitter.get_splits(df):
            X_train = train_split.drop(columns=["y"]).to_numpy()
            y_train = np.eye(10)[train_split["y"].to_list()]
            X_test = test_split.drop(columns=["y"]).to_numpy()
            y_test = np.eye(10)[test_split["y"].to_list()]

            epoch_loss = model.refit(X_train, y_train)
            losses = [l + e for l, e in zip(losses, epoch_loss)]
            predictions = model.predict(X_test)
            predictions = [z.reshape(-1) for z in predictions]
            pred_labels = np.argmax(predictions, axis=1)
            actual_labels = np.argmax(y_test, axis=1)

            avg_accuracy += accuracy(pred_labels, actual_labels)
            avg_f1_score += f1(pred_labels, actual_labels)

        losses = [l / 10 for l in losses]
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
            best_losses = losses

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
    
    plt.plot(list(range(len(best_losses))), best_losses, label="J over training epochs", marker="o")
    plt.ylim(min(best_losses) - 0.05, max(best_losses) + 0.05)
    plt.xlabel("Training Epochs")
    plt.ylabel("Regularized J")
    plt.legend()
    plt.grid()
    plt.savefig("ozel/figures/mnist_nn.png")
    plt.clf()


if __name__ == "__main__":
    run_nn()
    # run_rf()
    # run_knn()
