import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from flare.preprocessing import identify_columns, IGNORE_TYPES


def set_sns(font_scale=1.2, **kwargs):
    sns.set(font_scale=font_scale, **kwargs)


def fit_kmeans(data, k=2):
    """Fit K-Means on a dataset.

    Parameters
    ----------
    k: int, default=2
        The number of clusters to find.

    Returns
    -------
    inertia: float
        The within-cluster variance.
    """
    kmeans = KMeans(n_clusters=k).fit(data)
    return kmeans.inertia_


def plot_elbow(data, max_k=15, verbose=True):
    """Plot an elbow-plot of the within-cluster variance for a range of clusterings.

    Parameters
    ----------
    max_k: int, default=2
        The maximum number of clusters to try.

    Returns
    -------
    fig: matplotlib.Figure
        The figure.
    ax: matplotlib.Axes
        The axis.
    """
    errors = [np.nan, np.nan]
    for k in range(2, max_k + 1):
        if verbose:
            print("\rFitting for k={}".format(k), end="\n" if max_k == k else "")
        error = fit_kmeans(data, k=k)
        errors.append(error)

    if verbose:
        print("Done fitting, plotting elbow.")

    set_sns()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax = sns.lineplot(np.arange(max_k + 1, dtype=int), errors, marker="o", ax=ax)
    ax.set_ylabel("within-cluster variance")
    ax.set_xlabel("number of clusters")
    ax.set_title("Elbow plot for number of clusters", weight="bold")
    ax.set_xticks([x for x in range(0, max_k, 2)])
    fig.tight_layout()
    return fig, ax


def plot_tsne(tsne_data, color_labels=None, title="t-SNE plot of found clusters",
              figsize=(6, 5)):
    """PLot a t-SNE plot based on pre-calculated t-SNE data.

    Parameters
    ----------
    tsne_data: np.array
        The data to plot, usually output of sklearn.manifold. Is assumed to have two columns/variables, which will be plotted
        against each other in a scatter plot.
    color_labels: array-like, default=None
        Optional labels of length len(tsne_data) specifying which points to give the same
        color. Useful to plot calculated clusters.

    Returns
    -------
    fig: matplotlib.Figure
        The figure.
    ax: matplotlib.Axes
        The axis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=color_labels, ax=ax,
                         palette="Set3", legend=False)
    ax.set_title(title, weight="bold", size=16)
    ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    ax.set_xlabel("t-SNE feature 1")
    ax.set_ylabel("t-SNE feature 2")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.tight_layout()
    return fig, ax


def plot_cluster_means(data, id_col="C28992R100", cluster_col="cluster",
                       feat_colname="feature", value_colname="value", title="Cluster means",
                       col_wrap=2, aspect=3.5, xaxis_at_zero=True, top=0.9, wspace=None,
                       bottom=None, title_size=16):
    """Plot multiple bar plots to inspect the variable means of clusters.

    Parameters
    ----------
    data: pd.DataFrame
    id_col: str, default="c28992R100"
        Column that identifies instances.
    cluster_col: str, default="cluster"
        Column that specifies the cluster of each instance.
    feat_colname: str, default="feature"
        How to name the long-format column specifiying the variables.
    value_colname: str, default="value"
        How to name the long-format column specifiying the values.
    title: str, default="Cluster means"
        The title of the plot.
    col_wrap: int, default=2
        How many subplots to fit on one row.
    aspect: float, default=3.5
        The width / height ratio of each subplot.
    xaxis_at_zero: bool, default=True
        Whether to put the xaxis at zero, rather than at the bottom.
    top, wspace, bottom: float, defaults=[0.9, None, None]
        How much space to reserve for the title on top: 1 - top = title space
        in between subplots horizontally and at the bottom for the tick and
        axis labels.
    title_size: int, default=16
        The font size of the title.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        The figure.
    """
    # to long format
    data = data.copy()
    data = data.set_index([cluster_col, id_col]).stack().reset_index()
    data.columns = [cluster_col, id_col, feat_colname, value_colname]
    variable_order = np.unique(data[feat_colname])
    # plot
    set_sns()
    g = sns.FacetGrid(data, col=cluster_col, col_wrap=col_wrap, height=2, aspect=aspect)
    g.map(sns.barplot, feat_colname, value_colname, order=variable_order)
    
    if xaxis_at_zero:
        for ax in g.axes:
            ax.axhline(0, color="k", clip_on=False)
        sns.despine(bottom=True)

    for ax in g.axes:
        ax.tick_params(axis='x', which='both', length=0)

    g.set_xticklabels(rotation=90)
    g.fig.suptitle(title, size=title_size, weight="bold")
    g.fig.subplots_adjust(top=top, wspace=wspace, bottom=bottom)
    return g.fig


def add_number_of_incidents(data, path="../data/incidents.csv", id_col="C28992R100"):
    """Add a column with the total number of incidents to a DataFrame."""
    incident_data = pd.read_csv("D:/FireData/flare/grid_and_incidents_by_year.csv")
    types, features, _ = identify_columns(incident_data)
    types = [t for t in types if t not in IGNORE_TYPES]

    incident_data["number of incidents"] = incident_data[types].sum(axis=1)
    num_incidents = incident_data.groupby(id_col)["number of incidents"].mean().reset_index()
    data = pd.merge(data, num_incidents, on=[id_col], how='left')
    return data


def plot_number_of_incidents_per_cluster(data, cluster_col="demographics cluster",
                                         y_col="number of incidents", figsize=(8, 5),
                                         ylim=(0, 10), title="Number of incidents by cluster"):
    """Plot the distribution of the number of incidents per cluster."""
    set_sns()
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(x=cluster_col, y=y_col, data=data, ax=ax, meanline=True)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(title, weight="bold", size=16)
    fig.tight_layout()
    return fig
