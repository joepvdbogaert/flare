import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class AveragePredictor():
    """Naive estimator that predicts the average over previous timestamps for a given instance.

    Parameters
    ----------
    id_col: str
        The column in the data that acts as an identifier for an instance. This identifier must
        refer to the same instance, but at different time moments, in the train and test set.
    """

    def __init__(self, id_col=None):
        self.id_col = id_col

    def fit(self, X, Y):
        Y = pd.Series(Y, name="target").copy()
        X = pd.DataFrame(X).copy()
        self.data = pd.concat([X, Y], axis=1)
        self.averages = self.data.groupby(self.id_col)["target"].mean()

    def predict(self, X):
        ids = X[self.id_col].values
        return self.averages.loc[ids].values

    def predict_proba(self, X):
        return self.predict(X)


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


def score_for_threshold(y, y_hat, score_func, threshold):
    """Calculate the score on prediction-probabilities given a decision threshold.

    Parameters
    ----------
    y, y_hat: np.arrays
        The true label and predictions respectively.
    score_func: callable
        The function to compute the score with.
    threshold: float
        The decision treshold. All predictions above threshold will be predicted as
        1, the rest as 0.

    Returns
    -------
    score: float
        The score.
    """
    y_rounded = np.where(y_hat >= threshold, 1, 0)
    return scoring(y, y_rounded)


def find_best_threshold(y, y_hat, step_size, score_func, maximize=True):
    """Calculate the  decision threshold that leads to the best score.

    Parameters
    ----------
    y, y_hat: np.arrays
        The true label and predictions respectively.
    step_size: float
        The granularity of the threshold to consider. E.g. when step_size=0.01,
        tries 100 thresholds: 0.0, 0.01, 0.02, ..., 0.99.
    score_func: callable
        The function to compute the score with.
    threshold: float
        The decision treshold. All predictions above threshold will be predicted as
        1, the rest as 0.
    maximize: bool, default=True
        Whether to return the threshold that maximizes (True) or minimizes (False)
        the score.

    Returns
    -------
    threshold: float
        The best threshold
    score: float
        The score corresponding to the best threshold.
    """
    best_thres, best_score = 0.0, 0.0 if maximize else 1.0
    for thres in np.arange(0, 1, step_size):
        score = score_for_threshold(y, y_hat, score_func, thres)
        if (maximize and (score > best_score)) or (not maximize and (score < best_score)):
            best_score = score
            best_thres = thres

    return best_thres, best_score


def best_threshold_from_folds(y_tuples, scoring=f1_score, step_size=0.01, maximize=True):
    """Calculate the optimal decision treshold based on (multiple sets of)
    probability predictions and true labels.

    Parameters
    ----------
    y_tuples: list(tuple(array, array))
        List of (y_true, y_pred) tuples, where y_true and y_pred are arrays.
    score_func: callable, default=sklearn.metrics f1_score
        The function to compute the score with.
    step_size=float, default=0.01
        Granularity of the threshold.
    maximize: bool, default=True
        Whether to return the threshold that maximizes (True) or minimizes (False)
        the score.

    Returns
    -------
    threshold: float
        The mean best decision threshold over the folds.
    score: float
        The mean score.
    """
    thresholds, scores = [], []
    for y_true, y_pred in y_tuples:
        t, s = find_best_threshold(y_true, y_pred, step_size, scoring, maximize=maximize)
        thresholds.append(t)
        scores.append(s)

    mean_threshold = np.mean(thresholds)
    mean_score = np.mean([score_for_threshold(y, y_hat, scoring, mean_threshold) for y, y_hat in y_tuples])
    return mean_threshold, mean_score
