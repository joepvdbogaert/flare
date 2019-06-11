import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


class YearSplitter():
    """Class that splits data in train and test sets by year.

    The splits are constructed in chronological order, where every test set is one year and
    the preceding years are the corresponding train data.

    Parameters
    ----------
    folds: int
        The number of folds to use when splitting the data.
    """
    def __init__(self, folds=4, year_col="YEAR"):
        self.folds = folds
        self.year_col = year_col

    def split(self, data):
        """Split the data into train and test sets over `self.folds` number of folds.

        Parameters
        ----------
        data: pd.DataFrame
            The data to split.
        year_col: str, default='YEAR'
            The column in data that specifies the year.

        Yields
        -------
        (train, test): pd.DataFrames
            Tuples of train and test splits.
        """
        years = np.sort(np.unique(data[self.year_col]))
        assert len(years) >= self.folds + 1, ("Cannot fold {} times with only {} years in"
                                              " the data".format(self.folds, len(years)))

        splits = [years[len(years) - i] for i in range(1, self.folds + 1)]

        for threshold in splits[::-1]:
            yield data[data[self.year_col] < threshold], data[data[self.year_col] == threshold]


def evaluate_model(model, x, y, scoring, score_on_proba=False):
    """Evaluate a trained model and return the score.

    Parameters
    ----------
    model: object
        Trained estimator that implements the `model.predict(x)` method.
    x: 2D array-like,
        The features to use for prediction.
    y: 1D array-like
        The true labels / values corresponding to x.
    scoring: function
        Used to calculate the score. Must take true_label, predicted_label as inputs.
    """
    if score_on_proba:
        y_hat = model.predict_proba(x)
        if y_hat.shape[1] == 2:
            # binary prediction, keep proba of 1
            y_hat = y_hat[:, 1].flatten()
    else:
        y_hat = model.predict(x)
    score = scoring(y, np.asarray(y_hat, dtype=np.float))
    return score

def cross_validate_by_year(model_cls, data, x_cols, y_col, model_params=None, folds=4,
                           year_col="YEAR", scoring=mean_squared_error, return_all=False,
                           score_on_proba=False, verbose=True):
    """Cross validate using different years as validation and training sets.

    Parameters
    ----------
    model_cls: Python class
        The predictor class to use. Must implement the `.fit` and `.evaluate` methods.
    data: pd.DataFrame
        The data to train and validate on.
    x_cols: list(str)
        The columns in the data to use as features in prediction.
    y_col: str
        The column in the data to predict.
    model_params: dict
        Parameters to pass to the model class upon initialization.
    folds: int, default=4
        The number of folds to use in cross validation.

    Returns
    -------
    scores: list(float)
        The errors on the validation data for each fold.
    """
    if model_params is None:
        model_params = {}

    splitter = YearSplitter(folds=folds, year_col=year_col)

    train_scores = []
    val_scores = []
    for i, (train, val) in enumerate(splitter.split(data)):
        # select features and targets
        train_x = train[x_cols]
        train_y = train[y_col]
        val_x = val[x_cols]
        val_y = val[y_col]

        # train
        model = model_cls(**model_params)
        model.fit(train_x, train_y)

        # evaluate on training data
        train_scores.append(evaluate_model(model, train_x, train_y, scoring, score_on_proba=score_on_proba))
        val_scores.append(evaluate_model(model, val_x, val_y, scoring, score_on_proba=score_on_proba))
        if verbose:
            print("Fold {}. train score: {}, val score: {}".format(i + 1, train_scores[i], val_scores[i]))

    if return_all:
        return val_scores
    else:
        return np.mean(val_scores)
