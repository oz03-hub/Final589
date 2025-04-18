import pandas as pd
import numpy as np
from collections import Counter
import math
import random
import multiprocessing as mp


class LeafNode:
    def __init__(self, y):
        self.type = "leaf"
        self.y = y


class CategoricalDecisionNode:
    def __init__(self, attribute: str, edges: dict, majority_class: str):
        self.type = "cat_decision"
        self.attribute = attribute
        self.edges = edges
        self.majority_class = majority_class


class NumericDecisionNode:
    def __init__(
        self, attribute: str, split_point: float, edges: dict, majority_class: str
    ):
        self.type = "num_decision"
        self.attribute = attribute
        self.split_point = split_point
        self.edges = edges
        self.majority_class = majority_class


class RandomDecisionTree:
    def __init__(
        self,
        bagged_data: pd.DataFrame,
        m=None,
        stopping_criterion="minimal_gain",
        stopping_criterion_hyperparameter=0.01,
        class_column="label",
        numeric_end="_num",
    ):
        """stopping_criterion can be minimal_gain, minimal_size_for_split, or maximal_depth"""

        self.bagged_data: pd.DataFrame = bagged_data
        self.class_column: str = class_column
        self.attributes: list[str] = bagged_data.columns.drop(
            self.class_column
        ).to_list()
        if m is None:
            self.m: int = int(math.sqrt(len(bagged_data.columns)))
        else:
            self.m: int = m

        self.stopping_criterion: str = stopping_criterion
        self.stopping_criterion_hyperparameter: float = (
            stopping_criterion_hyperparameter
        )
        self.numeric_end: str = numeric_end  # numeric attributes end with this string

        self.tree = self._fit_recursively(self.bagged_data)  # train on creation

    def predict(self, x: pd.Series):
        # HW1 Code
        node = self.tree
        while node.type != "leaf":
            if node.type == "cat_decision":  # if categorical check value match
                edge_value = x[node.attribute]
                if edge_value in node.edges:
                    node = node.edges[edge_value]
                else:
                    return node.majority_class
            elif node.type == "num_decision":
                edge_value = x[node.attribute]
                if edge_value <= node.split_point:  # if numeric check if lesser
                    node = node.edges["lesser"]
                else:
                    node = node.edges["greater"]
        return node.y  # reached a leaf node

    # define helper functions
    def _entropy(self, y: pd.Series) -> float:
        # HW1 implementation, replaced counter with numpy function
        labels, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _id3(self, df: pd.DataFrame, attribute_to_split: str) -> tuple[any, float]:
        """
        Returns (split_point?, gain)
        """
        # HW1 Implementation + numeric handling
        og_entropy = self._entropy(df[self.class_column])

        if attribute_to_split.endswith(self.numeric_end):  # numeric attribute
            split_point = df[attribute_to_split].mean()
            df_left = df[df[attribute_to_split] <= split_point]
            df_right = df[df[attribute_to_split] > split_point]
            e_left = self._entropy(df_left[self.class_column])
            e_right = self._entropy(df_right[self.class_column])
            avg_entropy = (len(df_left) / len(df)) * e_left + (
                len(df_right) / len(df)
            ) * e_right
            gain = og_entropy - avg_entropy
            return split_point, gain
        else:  # categorical attribute
            # HW1 Code
            attribute_values = np.unique(df[attribute_to_split])
            splits = {
                value: df[df[attribute_to_split] == value] for value in attribute_values
            }
            avg_entropy = sum(
                len(split) / len(df) * self._entropy(split[self.class_column])
                for split in splits.values()
            )
            return None, og_entropy - avg_entropy

    def _fit_recursively(self, df: pd.DataFrame, depth: int = 0):
        # HW1 Code + extra stopping criteria
        depth += 1

        # BASE CASE: all same class
        if all(df[self.class_column] == df[self.class_column].iloc[0]):
            return LeafNode(
                df[self.class_column].iloc[0]
            )  # Create Leaf node with label

        # find majority class
        counts = Counter(df[self.class_column])
        majority_class = counts.most_common(1)[0][0]

        # BASE CASE: reached maximal depth, if set
        if (
            self.stopping_criterion == "maximal_depth"
            and depth >= self.stopping_criterion_hyperparameter
        ):
            return LeafNode(majority_class)

        # BASE CASE: minimal size for split
        if (
            self.stopping_criterion == "minimal_size_for_split"
            and len(df) < self.stopping_criterion_hyperparameter
        ):
            return LeafNode(majority_class)

        # select random m attributes
        attributes_to_consider = random.sample(self.attributes, self.m)

        # find best attribute to split
        attribute_gains: dict[str, tuple[any, float]] = {
            attribute: self._id3(df, attribute) for attribute in attributes_to_consider
        }
        best_gain_entry: tuple[str, tuple[any, float]] = max(
            attribute_gains.items(), key=lambda x: x[1][1]
        )  # find max gain

        best_attribute = best_gain_entry[0]
        split_point = best_gain_entry[1][0]  # None if categorical, only use in numeric
        best_gain = best_gain_entry[1][1]

        # BASE CASE: reached stopping criteria, minimal gain
        if (
            self.stopping_criterion == "minimal_gain"
            and best_gain < self.stopping_criterion_hyperparameter
        ):
            return LeafNode(majority_class)

        # create decision node
        if best_attribute.endswith(self.numeric_end):
            node = NumericDecisionNode(best_attribute, split_point, {}, majority_class)

            left_df = df[df[best_attribute] <= split_point]
            right_df = df[df[best_attribute] > split_point]

            # process lesser edge
            if len(left_df) > 0:
                node.edges["lesser"] = self._fit_recursively(left_df, depth)
            else:
                node.edges["lesser"] = LeafNode(majority_class)  # BASE CASE

            # process greater edge
            if len(right_df) > 0:
                node.edges["greater"] = self._fit_recursively(right_df, depth)
            else:
                node.edges["greater"] = LeafNode(majority_class)  # BASE CASE

            return node
        else:
            # HW1 Code
            node = CategoricalDecisionNode(best_attribute, {}, majority_class)

            # fill edges recursively
            for value in np.unique(df[best_attribute]):
                split_df = df[df[best_attribute] == value]
                if len(split_df) > 0:
                    node.edges[value] = self._fit_recursively(split_df, depth)
                else:
                    node.edges[value] = LeafNode(majority_class)  # BASE CASE

            return node


class RandomForest:
    def __init__(
        self,
        n_tree: int,
        data: pd.DataFrame,
        class_column: str = "label",
        m: int = None,
        stopping_criterion: str = "minimal_gain",
        stopping_criterion_hyperparameter: float = 0.01,
        numeric_end: str = "_num",
        multiprocessing: bool = True,
    ):
        """stopping_criterion can be minimal_gain, minimal_size_for_split, or maximal_depth"""
        self.n_tree = n_tree
        self.data = data
        self.class_column = class_column
        self.m = m
        self.stopping_criterion = stopping_criterion
        self.stopping_criterion_hyperparameter = stopping_criterion_hyperparameter
        self.numeric_end = numeric_end
        self.multiprocessing = multiprocessing
        self.trees = self._fit()  # fit on creation

    def _fit(self):
        boots = self._create_boots()
        if self.multiprocessing:
            # don't use multiprocessing if you have a bad cpu or not enough ram
            # source: https://docs.python.org/3/library/multiprocessing.html
            # used the first map implementation
            with mp.Pool(
                processes=mp.cpu_count() - 2
            ) as pool:  # it will try not to abuse your cpu
                trees = pool.map(self._fit_tree, boots)
            return trees
        else:
            trees = [self._fit_tree(boot) for boot in boots]
            return trees

    def _fit_tree(self, boot: pd.DataFrame):
        return RandomDecisionTree(
            boot,
            m=self.m,
            stopping_criterion=self.stopping_criterion,
            stopping_criterion_hyperparameter=self.stopping_criterion_hyperparameter,
            numeric_end=self.numeric_end,
            class_column=self.class_column,
        )

    def _create_boots(self):
        boots = []
        for _ in range(self.n_tree):
            boot = self.data.sample(
                n=len(self.data), replace=True
            )  # bootstrap, with replacement
            boots.append(boot)
        return boots

    def predict(self, x: pd.Series):
        predictions = [tree.predict(x) for tree in self.trees]  # majority voting
        counts = Counter(predictions)

        # if there is a tie, return a random choice
        if (
            len(counts.items()) > 1
            and counts.most_common(1)[0][1] == counts.most_common(2)[1][1]
        ):
            return random.choice(list(counts.keys()))
        else:
            return counts.most_common(1)[0][0]

    def confusion_matrix(self, X: pd.DataFrame, label_map: dict[str, int] = None):
        # HW2 Code
        """
        Returns TP, FP, FN, TN
        """

        predictions = X.apply(self.predict, axis=1)
        actual = X[self.class_column]

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        if label_map:
            predictions = predictions.map(label_map)
            actual = actual.map(label_map)

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


# old test code, using tennis set from lecture
# if __name__ == "__main__":
# df = pd.read_csv("tennis.csv")
# dt = RandomDecisionTree(df)
# for i in range(len(df)):
#     print(dt.predict(df.iloc[i]))
