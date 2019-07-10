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


def optimal_threshold(y_tuples, scoring=f1_score, step_size=0.01, maximize=True):
    """Calculate the optimal decision treshold based on (multiple sets of)
    probability predictions and true labels.

    Parameters
    ----------
    y_tuples: list(tuple(array, array))
        List of (y_true, y_pred) tuples, where y_true and y_pred are arrays.
    step_size=float, default=0.01
        Granularity of the threshold.

    Returns
    -------
    threshold: float
        The mean best decision threshold over the folds.
    score: float
        The mean score.
    """
    def score_for_threshold(y, y_hat, score_func, threshold):
        y_rounded = np.where(y_hat >= threshold, 1, 0)
        return scoring(y, y_rounded)

    def find_best(y, y_hat, step_size, score_func, maximize=True):
        best_thres, best_score = 0.0, 0.0 if maximize else 1.0
        for thres in np.arange(0, 1, step_size):
            score = score_for_threshold(y, y_hat, score_func, thres)
            if (maximize and (score > best_score)) or (not maximize and (score < best_score)):
                best_score = score
                best_thres = thres

        return best_thres, best_score

    thresholds, scores = [], []
    for y_true, y_pred in y_tuples:
        t, s = find_best(y_true, y_pred, step_size, scoring, maximize=maximize)
        thresholds.append(t)
        scores.append(s)

    mean_threshold = np.mean(thresholds)
    mean_score = np.mean([score_for_threshold(y, y_hat, scoring, mean_threshold) for y, y_hat in y_tuples])
    return mean_threshold, mean_score
