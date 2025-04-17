import pandas as pd
from kfold import StratifiedKFold
from itertools import product
from random_forest import RandomForest

if __name__ == "__main__":
    # Random Forest for Credit Approval Dataset
    credit_df = pd.read_csv("data/credit_approval.csv")
    splitter = StratifiedKFold(k=10)

    hyperparams = {
        "n_tree": [2, 5, 10, 20, 30, 50],
    }

    param_grid = product(*hyperparams.values())
    param_names = list(hyperparams.keys())

    best_params = None
    best_accuracy = 0
    best_f1_score = 0

    print("Random Forest for Credit Approval Dataset")
    for params in param_grid:
        params_dict = dict(zip(param_names, params))

        avg_accuracy = 0
        avg_f1_score = 0

        for train_split, test_split in splitter.get_splits(credit_df):
            rf_model = RandomForest(n_tree=params_dict["n_tree"], data=train_split)
            tp, fp, fn, tn = rf_model.confusion_matrix(test_split)
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
