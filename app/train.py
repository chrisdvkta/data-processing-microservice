import pandas as pd


def test():
    dataset = pd.read_csv("data/test.csv")
    print(dataset)
