import numpy as np
from collections import Counter
import random

random.seed(4)


class StandardNB:
    def __init__(self, pos: list[list[str]], neg: list[list[str]], vocab: list[str]):
        self.positive_instances = pos
        self.negative_instances = neg
        self.vocab = vocab
        self._fit()

    def _transform_to_tf_dict(self):
        # each key a term, each value a count, no matrix blew up the memory

        # save n(w_s, y_i) for each class
        self.pos_tf_dict = {}
        for doc in self.positive_instances:
            tf = Counter(doc)
            for word, f in tf.items():
                if word not in self.pos_tf_dict:
                    self.pos_tf_dict[word] = f
                else:
                    self.pos_tf_dict[word] += f

        self.neg_tf_dict = {}
        for doc in self.negative_instances:
            tf = Counter(doc)
            for word, f in tf.items():
                if word not in self.neg_tf_dict:
                    self.neg_tf_dict[word] = f
                else:
                    self.neg_tf_dict[word] += f

    def _fit(self):
        self._transform_to_tf_dict()

        # calculate p(y_i)
        self.p_y_pos = len(self.positive_instances) / (
            len(self.positive_instances) + len(self.negative_instances)
        )
        self.p_y_neg = len(self.negative_instances) / (
            len(self.positive_instances) + len(self.negative_instances)
        )

        self.pos_n = sum(
            self.pos_tf_dict.values()
        )  # calculate sum of n(w_s, y_i) for each class
        self.neg_n = sum(self.neg_tf_dict.values())

    def predict(self, doc: list[str]):
        p_p = self.p_y_pos  # multiply starts with p(y_i)
        p_n = self.p_y_neg

        doc = set(doc)  # unique words in doc

        for word in doc:
            if word not in self.vocab:
                p_p *= 0
                p_n *= 0
            else:
                p_p *= self.pos_tf_dict.get(word, 0) / self.pos_n
                p_n *= self.neg_tf_dict.get(word, 0) / self.neg_n

        # choose negative class if both same, 0
        if p_p > p_n:
            return 1  # pos class
        else:
            return 0  # neg class

    def predict_set(self, doc_set: list[tuple[list[str], int]]):
        return [self.predict(doc[0]) for doc in doc_set]  # ignore the label

    def build_confusion_matrix(self, doc_set_with_labels: list[tuple[list[str], int]]):
        predictions = self.predict_set(doc_set_with_labels)
        labels = [doc[1] for doc in doc_set_with_labels]

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for prediction, label in zip(predictions, labels):
            if prediction == 1 and label == 1:
                tp += 1
            elif prediction == 0 and label == 0:
                tn += 1
            elif prediction == 1 and label == 0:
                fp += 1
            elif prediction == 0 and label == 1:
                fn += 1

        return tp, tn, fp, fn
