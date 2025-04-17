import pandas as pd
import numpy as np
from collections import Counter


def normalize_data(df) -> pd.DataFrame:
    for column in df.columns:
        max_value = df[column].max()
        min_value = df[column].min()
        df[column] = (df[column] - min_value) / (max_value - min_value)
    return df


class KNN:
    """Operates only on numeric data. Supports multi-class classification. Don't forget to normalize the data before fitting."""

    def __init__(self, k: int):
        """
        Args:
            k: Number of neighbors to consider.
        """
        self.k = k

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Args:
            X_train: MUST BE DATAFRAME.
            y_train: MUST BE SERIES.
        """
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()

    def _euclidean_distances(self, x: np.ndarray, X_list: np.ndarray) -> np.ndarray:
        differences = (x - X_list) ** 2
        return np.sqrt(np.sum(differences, axis=1))

    def _knn_predict(self, X_unk: pd.Series) -> int:
        distances = self._euclidean_distances(X_unk.to_numpy(), self.X_train)
        distances = list(zip(distances, self.y_train))
        distances.sort(key=lambda x: x[0])

        k_top = distances[: self.k]
        counts = Counter([t[1] for t in k_top])
        max_class = counts.most_common(1)[0][0]

        return max_class

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Args:
            X_test: MUST BE DATAFRAME.
        """
        return X_test.apply(self._knn_predict, axis=1)

    def confusion_matrix(
        self, X_test: pd.DataFrame, y_test: pd.Series, label_map: dict[str, int] = None
    ) -> tuple[int, int, int, int]:
        """
        Args:
            X_test: MUST BE DATAFRAME.
            y_test: MUST BE SERIES.
            label_map: Optional dictionary to map labels to integers. Use for rice dataset.
        """
        predictions = self.predict(X_test)
        actual = y_test

        if label_map:
            predictions = predictions.map(label_map)
            actual = actual.map(label_map)

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for p, a in list(zip(predictions, actual)):
            if p == 1 and a == 1:
                tp += 1
            elif p == 1 and a == 0:
                fp += 1
            elif p == 0 and a == 1:
                fn += 1
            elif p == 0 and a == 0:
                tn += 1

        return tp, fp, fn, tn
