import numpy as np
import pandas as pd


class AveragePredictor():
    """Naive estimator that predicts the average over previous timestamps for a given instance.

    Parameters
    ----------
    id_col: str
        The column in the data that acts as an identifier for an instance. This identifier must
        refer to the same instance, but at different time moments, in the train and test set.
    """

    def __init__(self, id_col):
        self.id_col = id_col

    def fit(self, X, Y):
        Y = pd.Series(Y, name="target")
        X = pd.DataFrame(X)
        self.data = pd.concat([X, Y], axis=1)
        self.averages = self.data.groupby(self.id_col)["target"].mean()

    def predict(self, X):
        ids = X[self.id_col].values
        return self.averages.loc[ids]


class ZeroPredictor():

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.zeros(len(X))
