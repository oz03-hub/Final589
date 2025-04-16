import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
import math

# repeat steps a through e
# create two histograms, first one will show accuracy distribution of the dt when evaluated on training set
# second one shows accuracy on testing set
# train the algorithm 100 times: shuffling, splitting
# this will result in 100 training measures and 100 testing measures


def read_data(path):
    df = pd.read_csv(path, header=0, index_col=False)
    return df


def shuffle_data(df):
    return shuffle(df)


def split_data(df):
    return train_test_split(df, test_size=0.2)


def entropy(y):
    counts = Counter(y)
    probs = [count / len(y) for count in counts.values()]
    return sum(
        -prob * math.log2(prob) if prob > 0 else 0 for prob in probs
    )  # the 0log0 case pointed out


def id3(df: pd.DataFrame, attribute_to_split: str):
    og_entropy = entropy(df["class"])
    attribute_values = np.unique(df[attribute_to_split])
    splits = {}
    for value in attribute_values:
        splits[value] = df[df[attribute_to_split] == value]

    avg_entropy = sum(
        len(split) / len(df) * entropy(split["class"]) for split in splits.values()
    )
    return og_entropy - avg_entropy


def calc_gini(df: pd.DataFrame):
    counts = Counter(df["class"])
    probs = [count / len(df) for count in counts.values()]
    return 1 - sum(prob**2 for prob in probs)


def gini(df: pd.DataFrame, attribute_to_split: str):
    attribute_values = np.unique(df[attribute_to_split])
    splits = {}
    for value in attribute_values:
        splits[value] = df[df[attribute_to_split] == value]

    return sum(len(split) / len(df) * calc_gini(split) for split in splits.values())


def criterion(df: pd.DataFrame, attribute_to_split: str, type: str = "id3"):
    if type == "id3":
        return id3(df, attribute_to_split)
    elif type == "gini":
        return gini(df, attribute_to_split)


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


def decision_tree_recursive(
    df: pd.DataFrame,
    attributes: list,
    crit: str = "id3",
    heuristic_stopping: bool = False,
):
    # base case 1: all labels same
    if all(df["class"] == df["class"].iloc[0]):
        return LeafNode(df["class"].iloc[0])

    # base case 2: ran out of attributes
    if len(attributes) == 0:
        counts = Counter(df["class"])
        majority_class = counts.most_common(1)[0][0]
        return LeafNode(majority_class)

    # base case 3 (special): if more than 85% of instances belong to the same class
    if heuristic_stopping:
        counts = Counter(df["class"])
        mc_label, mc_count = counts.most_common(1)[0]
        if (mc_count / len(df["class"])) >= 0.85:
            return LeafNode(mc_label)

    if crit == "gini":
        best_attribute = min(attributes, key=lambda x: criterion(df, x, crit))
    elif crit == "id3":
        best_attribute = max(attributes, key=lambda x: criterion(df, x, crit))

    # calculate majority class at node
    counts = Counter(df["class"])
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
            edges[v] = decision_tree_recursive(
                df=d_v,
                attributes=new_attributes,
                crit=crit,
                heuristic_stopping=heuristic_stopping,
            )

    node.edges = edges
    return node


def decision_tree(
    df: pd.DataFrame, crit: str = "id3", heuristic_stopping: bool = False
):
    attributes = df.columns[:-1].to_list()
    return decision_tree_recursive(df, attributes, crit, heuristic_stopping)


def predict(tree, x: pd.Series):
    while tree.t == "decision":
        edge_value = x[tree.attribute]
        if edge_value in tree.edges:
            tree = tree.edges[edge_value]
        else:
            return tree.majority_class
    return tree.y


def accuracy(tree, df: pd.DataFrame):
    correct = 0
    for _, row in df.iterrows():
        if predict(tree, row) == row["class"]:
            correct += 1
    return correct / len(df)


# def print_tree(node):
#     if isinstance(node, LeafNode):
#         print(node.y)
#     else:
#         print(node.attribute)
#         for edge in node.edges.values():
#             print_tree(edge)


# print_tree(
#     decision_tree(
#         split_data(
#             shuffle(
#                 read_data("HW1_CMPSCI_589_Spring2025_Supporting_Files/datasets/car.csv")
#             )
#         )[0]
#     )
# )
