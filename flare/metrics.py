from sklearn.metrics import f1_score


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
