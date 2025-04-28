import pandas as pd
from kfold import StratifiedKFold
from knn import KNN, normalize_data
from itertools import product
from random_forest import RandomForest
from standard_nb import NaiveBayes
import matplotlib.pyplot as plt

def run_rf():
    # Random Forest for Rice Dataset
    rice_df = pd.read_csv("data/rice.csv")
    splitter = StratifiedKFold(k=10)

    hyperparams = {
        "n_tree": [2, 5, 10, 20, 30, 50],
    }

    param_grid = product(*hyperparams.values())
    param_names = list(hyperparams.keys())

    best_params = None
    best_accuracy = 0
    best_f1_score = 0

    print("Random Forest for Rice Dataset")
    accuracies = []
    f1_scores = []
    t= ""
    for params in param_grid:
        params_dict = dict(zip(param_names, params))

        avg_accuracy = 0
        avg_f1_score = 0
        for train_split, test_split in splitter.get_splits(rice_df):
            rf_model = RandomForest(n_tree=params_dict["n_tree"], data=train_split)
            tp, fp, fn, tn = rf_model.confusion_matrix(test_split, label_map={"Cammeo": 0, "Osmancik": 1})
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * precision * recall / (precision + recall)

            avg_accuracy += accuracy
            avg_f1_score += f1_score

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

    with open("ozel/results/rice_rf.txt", "w") as f:
        f.write(t)
        f.write("Best metrics'n")
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
    plt.savefig("ozel/figures/rice_rf.png")
    plt.clf()

def run_knn():
    # KNN for Rice Dataset
    rice_df = pd.read_csv("data/rice.csv")
    splitter = StratifiedKFold(k=10)

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
    print("KNN for Rice Dataset")
    for params in param_grid:
        params_dict = dict(zip(param_names, params))

        avg_accuracy = 0
        avg_f1_score = 0
        for train_split, test_split in splitter.get_splits(rice_df):
            X_train = normalize_data(train_split.drop(columns=["label"]))
            y_train = train_split["label"]
            X_test = normalize_data(test_split.drop(columns=["label"]))
            y_test = test_split["label"]

            knn_model = KNN(k=params_dict["k"])
            knn_model.fit(X_train, y_train)
            tp, fp, fn, tn = knn_model.confusion_matrix(
                X_test, y_test, label_map={"Cammeo": 0, "Osmancik": 1}
            )
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * precision * recall / (precision + recall)

            avg_accuracy += accuracy
            avg_f1_score += f1_score

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

    with open("ozel/results/rice_knn.txt", "w") as f:
        f.write(t)
        f.write("Best metrics\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Accuracy: {best_accuracy}\n")
        f.write(f"Best F1 Score: {best_f1_score}\n")

    plt.plot(hyperparams["k"], accuracies, label="Accuracy", marker="o")
    plt.plot(hyperparams["k"], f1_scores, label="F1 Score", marker="o")
    plt.ylim(min(*accuracies, *f1_scores) - 0.05, max(*accuracies, *f1_scores) + 0.05)
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy and F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig("ozel/figures/rice_knn.png")
    plt.clf()

def run_nb():
    # Naive Bayes for Rice Dataset
    rice_df = pd.read_csv("data/rice.csv")
    splitter = StratifiedKFold(k=10)

    hyperparams = {
        "alpha": [0.1, 0.5, 1, 2, 5, 10],
    }

    param_grid = product(*hyperparams.values())
    param_names = list(hyperparams.keys())

    best_params = None
    best_accuracy = 0
    best_f1_score = 0

    accuracies = []
    f1_scores = []
    t = ""
    print("Naive Bayes for Rice Dataset")
    for params in param_grid:
        params_dict = dict(zip(param_names, params))

        avg_accuracy = 0
        avg_f1_score = 0
        for train_split, test_split in splitter.get_splits(rice_df):
            nb_model = NaiveBayes(alpha=params_dict["alpha"])
            nb_model.fit(train_split.drop(columns=["label"]), train_split["label"])
            
            tp, fp, fn, tn = nb_model.confusion_matrix(test_split.drop(columns=["label"]), test_split["label"], label_map={"Cammeo": 0, "Osmancik": 1})
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * precision * recall / (precision + recall)

            avg_accuracy += accuracy
            avg_f1_score += f1_score

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

    with open("ozel/results/rice_nb.txt", "w") as f:
        f.write(t)
        f.write("Best metrics\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Accuracy: {best_accuracy}\n")
        f.write(f"Best F1 Score: {best_f1_score}\n")

    plt.plot(hyperparams["alpha"], accuracies, label="Accuracy", marker="o")
    plt.plot(hyperparams["alpha"], f1_scores, label="F1 Score", marker="o")
    plt.ylim(min(*accuracies, *f1_scores) - 0.05, max(*accuracies, *f1_scores) + 0.05)
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy and F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig("ozel/figures/rice_nb.png")
    plt.clf()

if __name__ == "__main__":
    run_rf()
    run_knn()
    run_nb()
