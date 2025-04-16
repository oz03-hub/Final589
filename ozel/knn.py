import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter

# shuffle dataset
# split 80% training, 20% testing
# use euclidean distance


def read_data(path):
    # return pd.read_csv(path, header=None)
    df = pd.read_csv(path, header=None, index_col=False)
    # print(df.head())
    return df


def normalize_data(df):
    for column in df.columns:
        max_value = df[column].max()
        min_value = df[column].min()
        df[column] = (df[column] - min_value) / (max_value - min_value)
    return df


def shuffle_data(df):
    return shuffle(df)


def split_data(df):
    """
    Returning: X_train, X_test, y_train, y_test
    """
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    # print("X: ", X.shape)
    # print("y: ", y.shape)

    return train_test_split(X, y, test_size=0.2)


def euclidean_distances(x, X_list):
    # print(x.shape) # (30, ) -> broadcast to (1, 30)
    # print(X_list.shape) # (455, 30)
    differences = (x - X_list) ** 2  # (455, 30)
    return np.sqrt(np.sum(differences, axis=1))  # (455, )


# compute accuracy of the knn model when used to make predictions for instances in the training set
# percentage of correct predictions made by the model when applied to the training data: correct / total
# compute same for testing set


def knn(path, k, norm=True):
    """
    Returning: acc_train, acc_test
    """
    df = read_data(path)
    if norm:
        df = normalize_data(df)

    X_train, X_test, y_train, y_test = split_data(shuffle_data(df))

    training_corrects = 0
    for x, y in zip(X_train, y_train):
        pred_class = knn_predict(x, X_train, y_train, k)

        if pred_class == y:
            training_corrects += 1

    training_accuracy = training_corrects / len(X_train)

    test_corrects = 0
    for x, y in zip(X_test, y_test):
        pred_class = knn_predict(x, X_train, y_train, k)

        if pred_class == y:
            test_corrects += 1
    test_accuracy = test_corrects / len(X_test)

    return training_accuracy, test_accuracy


def knn_predict(X_unk, X_train, y_train, k):
    """
    Runs for each predicted datapoint.

        Args: X_unk: single datapoint, X_train: list of datapoints
    """
    distances = euclidean_distances(X_unk, X_train)
    distances = list(zip(distances, y_train))
    distances.sort(key=lambda x: x[0])

    k_top = distances[:k]
    counts = Counter([t[1] for t in k_top])
    max_class = counts.most_common(1)[0][0]

    return max_class
