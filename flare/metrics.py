import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from flare.prediction import best_threshold_multiple_targets, proba_to_binary


def f1_macro(y_true, y_pred):
    """Compute the macro-averaged F1-score for given class labels and predictions.

    The macro-average is the average score over the labels. All classes thus have equal
    weight, no matter the proportion of samples in each.

    Parameters
    ----------
    y_true, y_pred: 1d-array
        The true and predicted class labels respectively.
    """
    return f1_score(y_true, y_pred, average='macro')


def classification_report_df(y_true, y_pred, target_name='y'):
    """Put sklearn's classification_report in a DataFrame.

    Parameters
    ----------
    y_true, y_pred: np.array
        True and predicted labels respectively.
    target_name: str, default='y'
        The target that was predicted. This is added as a column to
        facilitate putting multiple predictions in the same DataFrame.

    Returns
    -------
    report: pd.DataFrame
        The classification report.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    report = pd.DataFrame.from_dict(report).T
    report.index.name = 'label'
    report = report.reset_index()
    report["target"] = target_name
    return report


def classification_report_multiple_targets(data, cols, y_suff='_true', yhat_suff='_pred',
                                           thresholds=None):
    """Create a classification report with multiple metrics for multiple targets."""
    if thresholds is None:
        thresholds = best_threshold_multiple_targets(data, cols)

    reports = [classification_report_df(
                    data[c + y_suff],
                    proba_to_binary(data[c + yhat_suff].values, thresholds[c][0]),
                    target_name=c
               ) for c in cols]

    return pd.concat(reports, axis=0)


# function copied from sklearn examples
def plot_confusion_matrix(y_true, y_pred, classes, labels=None,
                          normalize=False, title=None,
                          cmap=plt.cm.Blues, figsize=(8, 8), rotate=False):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    if labels is None:
        labels = classes

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
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
           xticklabels=labels, yticklabels=labels,
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
