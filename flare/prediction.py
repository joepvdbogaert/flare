import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV


class OrdinalClassifier():
    """Wrapper class that turns any binary classifier into an Ordinal Classifier
    as proposed by Frank & Hall (2001).

    For K ordinal classes, the method fits K-1 binary classifiers that predict
    Pr(Y > y), y=1, ..., K-1. Inference is then performed by selecting
    Pr(Y = k) = Pr(Y > k - 1) * (1 - Pr(Y > k + 1)), for k=2, ..., K-1. For k=1
    and k=K, inference is trivial.

    Parameters
    ----------
    clf: sklearn classifier object, initialized
        Must implement the fit, predict, and predict_proba methods.
    calibration_data: tuple(2D array-like, 1D array-like)
        Data used to calibrate the confidence scores of individual classifiers. This
        calibrates the ouputted confidence scores, so that they are more aligned with
        the actual probability that the instance is of a given class. In other words,
        it makes this method have some sense.

    Notes
    -----
    The original method assumes confidence scores provided by the classifier can be interpreted
    as probabilities, which is highly doubtful in most cases. Frank and Hall make no notice
    of this in their paper. The option to calibrate the confidence scores was our idea and is
    not peer-reviewed or anything.
    
    The code is based on a blog post on TowardsDataScience (see references).

    References
    ----------
    [Frank and Hall (2001)](https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf)
    [Original code and blog post by Muhammad Assagaf](https:
    //towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
    """
    def __init__(self, clf, cal_data=None, cal_method='isotonic'):
        self.clf = clf
        self.clfs = {}
        self.cal_data = cal_data
        if cal_data is not None:
            self.x_cal, self.y_cal = cal_data
        if cal_method not in ['isotonic', 'sigmoid']:
            raise ValueError("cal_method must be one of ['isotonic', 'sigmoid']. Got {}."
                             .format(cal_method))
        self.cal_method = cal_method

    def fit(self, X, y):
        """Fit K-1 binary classifier to the data, where K is the number of unique values
        in y.

        Parameters
        ----------
        X: 2D array-like
            The input data.
        y: 1D array-like
            The ordinal target variable.
        """
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classifier
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                if self.cal_data is not None:
                    calib_clf = CalibratedClassifierCV(clf, cv='prefit', method=self.cal_method)
                    binary_y_cal = (self.y_cal > self.unique_class[i]).astype(np.uint8)
                    calib_clf.fit(self.x_cal, binary_y_cal)
                    self.clfs[i] = calib_clf
                else:
                    self.clfs[i] = clf

    def predict_proba(self, X):
        """Predict probabilities/confidence scores for each possible target value.

        Parameters
        ----------
        X: 2D array-like
            The input data.

        Returns
        -------
        yhat: 2D np.array
            The predicted probabilities of each class (axis 1) for each instance (axis 0).
        """
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])

        return np.vstack(predicted).T

    def predict(self, X):
        """Predict the target value for a given input.

        Parameters
        ----------
        X: 2D array-like
            The input data.

        Returns
        -------
        yhat: 1D np.array
            The predicted classes/values.
        """
        return np.argmax(self.predict_proba(X), axis=1)


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


def add_binary_indicator_per_type(data, cols, prefix="has_"):
    """Add binary variables for each column, indicating whether the column is bigger
    than zero (1) or not (0).

    Parameters
    ----------
    data: pd.DataFrame
        The data.
    cols: list(str)
        The columns for which to create an indicator.
    prefix: str
        What to put in front of the original column name to obtain a new column name. Set
        to '' (empty string) to store the result in the original column.

    Returns
    -------
    data: pd.DataFrame
        The data with added columns.
    binary_cols: list(str)
        The column names of the new columns.
    """
    binary_cols = ["{}{}".format(prefix, col) for col in cols]
    for i, col in enumerate(cols):
        data[binary_cols[i]] = np.array(data[col] > 0, dtype=np.int)
    return data, binary_cols


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
    return score_func(y, y_rounded)


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


def best_threshold_multiple_targets(data, cols, y_suff='_true', yhat_suff='_pred',
                                    step_size=0.01, score_func=f1_score, maximize=True):
    """Find the best thresholds for multiple target columns.

    Parameters
    ----------
    data: pd.DataFrame
        The data with validation predictions and true labels for all columns.
    cols: list(str)
        The columns that were predicted.
    y_suff, yhat_suff: str
        The suffix used to indicate the true and predicted version of the columns
        respectively.
    step_size: float
        The granularity of the threshold to consider. E.g. when step_size=0.01,
        tries 100 thresholds: 0.0, 0.01, 0.02, ..., 0.99.
    score_func: callable
        The function to compute the score with.
    maximize: bool, default=True
        Whether to return the threshold that maximizes (True) or minimizes (False)
        the score.

    Returns
    -------
    thresholds: dict
        The tresholds and corresponding scores in a dictionary like:
        {'col' -> (threshold, score)}.
    """
    thresholds = {
        c: find_best_threshold(
            data[c + y_suff].values,
            data[c + yhat_suff].values,
            step_size,
            score_func,
            maximize=maximize
        )
        for c in cols
    }
    return thresholds


def proba_to_binary(y_pred, threshold):
    return np.where(y_pred >= threshold, 1, 0)
