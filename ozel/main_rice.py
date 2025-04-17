import pandas as pd
from kfold import StratifiedKFold
from knn import KNN, normalize_data
from itertools import product
from random_forest import RandomForest

if __name__ == "__main__":
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

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_f1_score = avg_f1_score
            best_params = params_dict

    print(f"Best Params: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best F1 Score: {best_f1_score}")

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

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_f1_score = avg_f1_score
            best_params = params_dict

    print(f"Best Params: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best F1 Score: {best_f1_score}")
