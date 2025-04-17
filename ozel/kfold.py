import pandas as pd
from sklearn.utils import shuffle

class StratifiedKFold:
    def __init__(self, k: int = 5, class_column: str = "label"):
        self.k = k
        self.class_column = class_column

    def get_splits(self, X: pd.DataFrame) -> list[tuple]:
        # # shuffle, source: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        X = shuffle(X)

        labels = X[self.class_column].unique()
        label_instances = {
            label: len(X[X[self.class_column] == label]) // self.k for label in labels
        }

        # pos_instances: pd.DataFrame = X[X[self.class_column] == 1]
        # neg_instances: pd.DataFrame = X[X[self.class_column] == 0]

        # pos_size = len(pos_instances) // self.k
        # neg_size = len(neg_instances) // self.k

        # creating folds once
        folds = []
        for i in range(self.k):
            if (
                i == self.k - 1
            ):  # in case it is not divisible there are left overs, special case
                label_folds = []
                for label in labels:
                    label_folds.append(
                        X[X[self.class_column] == label].iloc[
                            i * label_instances[label] :
                        ]
                    )
                fold = pd.concat(label_folds)
            else:
                # get window of entries
                label_folds = []
                for label in labels:
                    label_folds.append(
                        X[X[self.class_column] == label].iloc[
                            i * label_instances[label] : (i + 1)
                            * label_instances[label]
                        ]
                    )
                fold = pd.concat(label_folds)

            folds.append(fold)

        train_test_splits = []
        for i in range(self.k):
            test_split = folds[i]
            train_split = pd.concat(
                [f for j, f in enumerate(folds) if j != i]
            )  # exclude test set

            test_split = test_split.reset_index(drop=True)  # got index errors
            train_split = train_split.reset_index(drop=True)
            train_test_splits.append((train_split, test_split))

        return train_test_splits


# old test code
# if __name__ == "__main__":
#     df = pd.read_csv("loan.csv")
#     print(df["label"].value_counts())
#     skf = StratifiedKFold(5, "label")
#     splits = skf.get_splits(df)
#     for train_split, test_split in splits:
#         print(train_split)
#         print(len(train_split))
#         print(train_split["label"].value_counts())
#         print(test_split)
#         print(len(test_split))
#         print(test_split["label"].value_counts())
#         print()
