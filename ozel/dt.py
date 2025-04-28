import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
import math

class LeafNode:
    def __init__(self, y):
        self.t = "leaf"
        self.y = y


class DecisionNode:
    def __init__(self, attribute: str, edges: dict, majority_class: str):
        self.t = "decision"
        self.attribute = attribute
        self.edges = edges
        self.majority_class = majority_class


class DecisionTree:
    def __init__(self, class_column: str = "label", crit = "id3"):
        self.class_column = class_column
        self.crit = crit
        self.tree = None

    def _entropy(self, y):
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return sum(
            -prob * math.log2(prob) if prob > 0 else 0 for prob in probs
        )  # the 0log0 case pointed out

    def _id3(self, df: pd.DataFrame, attribute_to_split: str):
        og_entropy = self._entropy(df[self.class_column])
        attribute_values = np.unique(df[attribute_to_split])
        splits = {}
        for value in attribute_values:
            splits[value] = df[df[attribute_to_split] == value]

        avg_entropy = sum(
            len(split) / len(df) * self._entropy(split[self.class_column]) for split in splits.values()
        )
        return og_entropy - avg_entropy

    def _calc_gini(self, df: pd.DataFrame):
        counts = Counter(df[self.class_column])
        probs = [count / len(df) for count in counts.values()]
        return 1 - sum(prob**2 for prob in probs)

    def _gini(self, df: pd.DataFrame, attribute_to_split: str):
        attribute_values = np.unique(df[attribute_to_split])
        splits = {}
        for value in attribute_values:
            splits[value] = df[df[attribute_to_split] == value]
        return sum(len(split) / len(df) * self._calc_gini(split) for split in splits.values())

    def _criterion(self, df: pd.DataFrame, attribute_to_split: str, type: str = "id3"):
        if type == "id3":
            return self._id3(df, attribute_to_split)
        elif type == "gini":
            return self._gini(df, attribute_to_split)

    def _fit_recursive(self, df, attributes):
        # base case 1: all labels same
        if all(df[self.class_column] == df[self.class_column].iloc[0]):
            return LeafNode(df[self.class_column].iloc[0])

        # base case 2: ran out of attributes
        if len(attributes) == 0:
            counts = Counter(df[self.class_column])
            majority_class = counts.most_common(1)[0][0]
            return LeafNode(majority_class)

        if self.crit == "gini":
            best_attribute = min(attributes, key=lambda x: self._criterion(df, x, self.crit))
        elif self.crit == "id3":
            best_attribute = max(attributes, key=lambda x: self._criterion(df, x, self.crit))

        # calculate majority class at node
        counts = Counter(df[self.class_column])
        majority_class = counts.most_common(1)[0][0]

        node = DecisionNode(best_attribute, {}, majority_class)
        edges = {}

        new_attributes = attributes[:]
        new_attributes.remove(best_attribute)

        V = np.unique(df[best_attribute])
        for v in V:
            d_v = df[df[best_attribute] == v]
            if len(d_v) == 0:
                edges[v] = LeafNode(majority_class)  # unreachable ?
            else:
                edges[v] = self._fit_recursive(
                    df=d_v,
                    attributes=new_attributes,
                )

        node.edges = edges
        return node


    def fit(self, df: pd.DataFrame):
        attributes = df.columns[:-1].to_list()
        root = self._fit_recursive(df, attributes)
        self.tree = root
        return self

    def predict(self, x: pd.Series):
        tree = self.tree
        while tree.t == "decision":
            edge_value = x[tree.attribute]
            if edge_value in tree.edges:
                tree = tree.edges[edge_value]
            else:
                return tree.majority_class
        return tree.y
