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
        if len(y_hat.shape) == 3:
            # received list of predictions, keep proba of 1 for each class
            y_hat = y_hat[:, :, 1]
            y_hat = y_hat.transpose()
        if y_hat.shape[1] == 2:
            # binary prediction, keep proba of 1
            y_hat = y_hat[:, 1].flatten()
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
                predictions.append((val_x, val_y, pd.Series(model.predict_proba(val_x), index=val_x.index)))
            else:
                predictions.append((val_x, val_y, pd.Series(model.predict(val_x), index=val_x.index)))
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


# function copied from sklearn examples
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None,
                          cmap=plt.cm.Blues, figsize=(8, 8), rotate=False):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_title(title, weight="bold", size=18)

    # Rotate the tick labels and set their alignment.
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
