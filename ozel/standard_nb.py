"""Improved smoothed NB code from HW2"""
import pandas as pd
import numpy as np
import math

class NaiveBayes:
    """Can fit categorical and numeric attributes."""
    def __init__(self, numeric_end: str = "_num", alpha: float = 0.1):
        """
        Args:
            numeric_end: The suffix of numeric attributes. No need to change usually.
            alpha: The smoothing parameter. Change to tune the model.
        """
        self.categorical_column_parameters = {}
        self.numeric_column_parameters = {}
        self.class_probabilities = {}
        self.numeric_end = numeric_end
        self.alpha = alpha
        self.classes = {}
        self.denoms = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Will fit to mixture of categorical and numeric attributes.

        Args:
            X_train: The training data. Must be a dataframe.
            y_train: The training labels. Must be a series.
        """

        for class_ in y_train.unique():
            if class_ not in self.numeric_column_parameters:
                self.numeric_column_parameters[class_] = {}
            if class_ not in self.categorical_column_parameters:
                self.categorical_column_parameters[class_] = {}

            self.classes[class_] = X_train[y_train == class_] # mask out rows of the class
            self.class_probabilities[class_] = len(self.classes[class_]) / len(X_train)
            self.denoms[class_] = {}

            for column in X_train.columns:                
                if column.endswith(self.numeric_end):
                    # calculate mean and std for numeric attributes
                    mean = self.classes[class_][column].mean()
                    std = self.classes[class_][column].std()
                    self.numeric_column_parameters[class_][column] = (mean, std)
                else:
                    # calculate probability for categorical attributes
                    self.categorical_column_parameters[class_][column] = self.classes[class_][column].value_counts()
                    self.denoms[class_][column] = len(self.classes[class_]) + self.alpha * len(self.classes[class_][column].unique())

    def _predict_row(self, row: pd.Series):
        class_probabilities = {}
        for class_ in self.classes:
            p = math.log(self.class_probabilities[class_])
            for column, value in row.items():
                if column.endswith(self.numeric_end):
                    mean, std = self.numeric_column_parameters[class_][column]
                    # gaussian estimation
                    estimation = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((value - mean) ** 2) / (2 * std ** 2)) # formula from lecture 6
                    if estimation != 0:
                        p += math.log(estimation)
                    else:
                        p += float("-inf")
                else:
                    vc = self.categorical_column_parameters[class_][column].get(value, 0)
                    p += math.log((vc + self.alpha) / self.denoms[class_][column])

            class_probabilities[class_] = p
        
        return max(class_probabilities, key=class_probabilities.get)

    def predict(self, X_test: pd.DataFrame):
        """
        Predicts the class of the test data.

        Args:
            X_test: The test data. Must be a dataframe.
        """
        return X_test.apply(self._predict_row, axis=1)

    def confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series, label_map: dict[str, int] = None):
        """
        Binary classification confusion matrix.

        Args:
            X_test: The test data. Must be a dataframe.
            y_test: The test labels. Must be a series.
            label_map: Optional dictionary to map labels to integers. Use for rice dataset.
        """
        predictions = self.predict(X_test)
        actual = y_test

        if label_map:
            predictions = predictions.map(label_map)
            actual = actual.map(label_map)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for prediction, label in zip(predictions, actual):
            if prediction == 1 and label == 1:
                tp += 1
            elif prediction == 0 and label == 0:
                tn += 1
            elif prediction == 1 and label == 0:
                fp += 1
            elif prediction == 0 and label == 1:
                fn += 1

        return tp, fp, fn, tn
