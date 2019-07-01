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


def add_incident_count_class_label(data, count_col="incidents", num_classes=6, one_hot=True):
    """Adds a column to the data with a class label of the number of incidents
    capped at a certain number.

    E.g., for num_classes = 3, the class labels become [0, 1, 2+].

    Parameters
    ----------
    data: pd.DataFrame
        The data to add the class labels to.
    count_col: str, default='incidents'
        The column that specifies the number of incidents.
    num_classes: int, default=6
        The number of classes to create.
    one_hot: bool, default=True
        Whether to add indicator columns for each class besides the 'class' column.

    Returns
    -------
    data: pd.DataFrame
        The data with the added column named 'class' and indicator columns if one_hot=True.
    """
    def add_plus(x, value=num_classes - 1):
        if int(x) == value:
            return str(x) + "+"
        return x

    data = data.copy()
    data["class"] = np.minimum(data[count_col].values, num_classes - 1)
    data["class"] = data["class"].astype(int).astype(str)
    data["class"] = data["class"].map(add_plus)

    # to onehot
    if one_hot:
        classes = np.sort(data["class"].unique())
        data = pd.concat([data, data["class"].str.get_dummies()], axis=1, ignore_index=False)
        class_labels = ["class_{}".format(x) for x in classes]
        data = data.rename(columns={x: "class_{}".format(x) for x in classes})
    
        return data, class_labels

    else:
        return data
