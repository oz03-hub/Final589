import pandas as pd
from kNN import generatekNNGraphs

if __name__ == "__main__":

    filePath = r'..\data\parkinsons.csv'
    generatekNNGraphs(fP=filePath)

