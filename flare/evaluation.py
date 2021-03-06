import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, confusion_matrix


def train_test_split_by_year(data, years_test=2, year_col="YEAR"):
    """Split a dataset in train and test by year.

    Parameters
    ----------
    data: pd.DataFrame
        The data to split.
    years_test: int, default=2
        The number of years to reserve for testing.
    year_col: str, default="YEAR"
        The column with the year.

    Returns
    -------
    train, test: pd.DataFrame
        The train and test sets respectively.
    """
    years = np.sort(np.unique(data[year_col]))
    cut = years[-years_test]
    print("Train contains [{}, {}], test set [{}, {}]."
          .format(years[0], cut - 1, cut, years[-1]))

    return data[data[year_col] < cut], data[data[year_col] >= cut]


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


def fix_shape(y_hat, to_pd=True, index=None):
    """Fix the shape of predictions originating from `predict_proba`."""
    if len(y_hat.shape) == 3:
        # received list of predictions, keep proba of 1 for each class
        y_hat = y_hat[:, :, 1]
        y_hat = y_hat.transpose()
    if len(y_hat.shape) > 1:
        if y_hat.shape[1] == 2:
            # binary prediction, keep proba of 1
            y_hat = y_hat[:, 1].flatten()

    # make pd.DataFrame or pd.Series if to_pd is True
    if to_pd:
        if len(y_hat.shape) > 1:
            y_hat = pd.DataFrame(y_hat, index=index)
        else:
            y_hat = pd.Series(y_hat, index=index)

    return y_hat


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
        y_hat = np.array(model.predict_proba(x))
        y_hat = fix_shape(y_hat, to_pd=False)
    else:
        y_hat = np.array(model.predict(x))

    score = scoring(y, y_hat)
    return score


def cross_validate_by_year(model_cls, data, x_cols, y_col, model_params=None, folds=4,
                           year_col="YEAR", scoring=mean_squared_error, return_all=False,
                           score_on_proba=False, verbose=True, pipe=False,
                           return_predictions=False):
    """Cross validate using different years as validation and training sets.

    Parameters
    ----------
    model_cls: Python class or sklearn.pipeline.Pipeline
        The predictor class to use. Must implement the `.fit` and `.evaluate` methods.
        If model_cls is a Pipeline, make sure to set pipe=True.
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
    year_col: str, default='YEAR'
        The column that specifies the year of the data points.
    scoring: callable, default=sklearn.metrics.mean_squared_error
        The function to use to calculate the score.
    return_all: bool, default=False
        If True, returns a list of scores for each fold. Otherwise returns the mean
        score over the folds. Ignored if return_predictions=True.
    score_on_proba: bool, default=False
        Whether to score on the probabilities or the class predictions. Calls `predict_proba`
        of the estimator if this is True, so this must be implemented in that case
    pipe: bool, default=False
        Set to true if model_cls is a pipeline instead of an uninitialized model. Note
        that model_params must be named accordingly (like: 'stepname__param').
    return_predictions: bool, default=False
        Return the predictions on the validation set of every fold instead of the scores.

    Returns
    -------
    scores: list(float) or list(tuple(np.array, np.array, np.array))
        The errors on the validation data for each fold. Unless return_predictions=True, then
        a list of tuples is returned like [(val_x_fold_1, val_y_fold_1, val_y_hat_fold_1), ...].
    """
    if model_params is None:
        model_params = {}

    splitter = YearSplitter(folds=folds, year_col=year_col)

    train_scores = []
    val_scores = []
    predictions = []  # only use if return_predictions=True
    for i, (train, val) in enumerate(splitter.split(data)):
        # select features and targets
        train_x = train[x_cols]
        train_y = train[y_col]
        val_x = val[x_cols]
        val_y = val[y_col]

        # train
        if pipe:
            model = model_cls.set_params(**model_params)
        else:
            model = model_cls(**model_params)

        model.fit(train_x, train_y)

        if return_predictions:
            if score_on_proba:
                predictions.append((val_x, val_y, fix_shape(model.predict_proba(val_x), to_pd=True, index=val_x.index)))
            else:
                predictions.append((val_x, val_y, fix_shape(model.predict(val_x), to_pd=True, index=val_x.index)))
        else:
            # evaluate on validation data and on training data if verbose
            val_scores.append(evaluate_model(model, val_x, val_y, scoring, score_on_proba=score_on_proba))
            if verbose:
                train_scores.append(evaluate_model(model, train_x, train_y, scoring, score_on_proba=score_on_proba))
                print("Fold {}. train score: {}, val score: {}".format(i + 1, train_scores[i], val_scores[i]))

    if return_predictions:
        return predictions
    elif return_all:
        return val_scores
    else:
        return np.mean(val_scores)


def cross_validate_multiple_targets(model_cls, data, feature_cols, target_cols, return_all=False,
                                    folds=4, verbose=True, return_predictions=False, **kwargs):
    """Cross validate multiple target columns indepently using the same model.

    Parameters
    ----------
    See `flare.evaluation.cross_validate_by_year`.

    Returns
    -------
    cv_results: pd.DataFrame
        The results with scores per incident type (row) and fold (column) if return_all=True,
        otherwise a single column with the mean score over folds.
    """
    if return_predictions:
        prediction_dict = {}
        for i, target in enumerate(target_cols):
            if verbose: print("\n{} (prop 1: {})\n--------------".format(target, np.mean(data[target] == 1)))
            prediction_dict[target] = cross_validate_by_year(
                model_cls, data, feature_cols, target,
                return_all=return_all, folds=folds, verbose=verbose,
                return_predictions=return_predictions, **kwargs
            )

        return prediction_dict
    else:  # return scores instead of predictions
        num_cols = folds if return_all else 1
        cv_results = pd.DataFrame(
            np.empty((len(target_cols), num_cols)),
            index=target_cols,
            columns=["fold {}".format(i) for i in np.arange(1, folds + 1)]
        )
        for i, target in enumerate(target_cols):
            if verbose: print("\n{} (prop 1: {})\n--------------".format(target, np.mean(data[target] == 1)))
            scores = cross_validate_by_year(
                model_cls, data, feature_cols, target,
                return_all=return_all, folds=folds, verbose=verbose,
                return_predictions=return_predictions, **kwargs
            )
            cv_results.iloc[i, :] = scores

        return cv_results


def construct_validation_data_from_folds(tuples, y_name="y"):
    """Reconstruct a DataFrame from cross validation predictions and input data.

    The function `flare.evaluation.cross_validate_by_year`, when ran with return_predictions=True,
    returns tuples of [(X_fold1, Y_fold1, Y_hat_fold1), ...]. This function joins these arrays into
    a pandas DataFrame again for further analysis.

    Parameters
    ----------
    tuples: list(tuples)
        Output of `flare.evaluation.cross_validate_by_year`: a list of (x, y, y_hat) tuples for
        every fold.
    y_name: str, default='y'
        The name of the target variable.

    Returns
    -------
    df: pd.DataFrame
        The reconstructed data set, where y, and y_hat are attached to x as columns. The y values have
        as colummn name y_name + '_true' and y_hat has y_name + '_pred'.
    """
    df = pd.concat(
        [pd.concat(
            [pd.DataFrame(x), pd.DataFrame({y_name+"_true": y, y_name + "_pred": y_hat}, index=x.index)],
            axis=1
        ) for (x, y, y_hat) in tuples],
        axis=0
    )
    return df


def construct_val_data_multiple_targets(pred_dict):
    """Reconstruct a validation dataset from cross validation predictions for multiple targets.

    The function `flare.evaluation.cross_validate_multiple_targets`, when ran with return_predictions=True,
    returns a dictionary with tuples of [(X_fold1, Y_fold1, Y_hat_fold1), ...] as values. This function joins
    these arrays into a pandas DataFrame again for further analysis.

    Parameters
    ----------
    tuples: dict(list(tuples))
        Output of `flare.evaluation.cross_validate_multiple_targets`: a dictionary of {'target' -> [(x, y, y_hat), ...]}
        where the lists have a tuple for every fold.

    Returns
    -------
    df: pd.DataFrame
        The reconstructed data set, where y, and y_hat are attached to x as columns. The y values have
        as colummn name y_name + '_true' and y_hat has y_name + '_pred'.
    """
    keys = list(pred_dict.keys())
    dfs = [construct_validation_data_from_folds(pred_dict[key], y_name=key)
           for key in keys]
    for i in range(len(dfs)):
        if i > 0:
            dfs[i] = dfs[i][[keys[i] + "_true", keys[i] + "_pred"]]

    return pd.concat(dfs, axis=1)


def construct_multiclass_val_data_probas(tuples, y_names=[str(x) for x in range(9)] + ['9+']):
    """Construct validation data when the output of predictions is 2D, such as in the
    multi-class classification case."""
    def set_cols(d, suffix='_true'):
        d.columns = [str(c) + suffix for c in y_names]
        return d

    # one hot encode true labels
    tuples = [(x, set_cols(pd.get_dummies(y)), set_cols(yhat, suffix='_proba')) for x, y, yhat in tuples]
    df = pd.concat(
        [pd.concat(
            [x, pd.DataFrame(y, index=x.index), pd.DataFrame(yhat, index=x.index)],
            axis=1
        ) for (x, y, yhat) in tuples],
        axis=0
    )
    return df
