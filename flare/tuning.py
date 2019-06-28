import numpy as np
import pandas as pd
import multiprocessing
import pickle

try:
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("Failed to import GPyOpt, either install it or refrain from using Bayesian Optimization")

from flare.evaluation import cross_validate_by_year


TUNING_OUTPUT_DEFAULT = '../results/tuning_output.pkl'


def print_eval_result(params, score, verbose=True):
    if verbose:
        print("--\nScore: {}\nParams: {}".format(score, params))


def bayesian_optimization(predictor_cls, data, x_cols, y_col, params, max_iter=250, max_time=None,
                          model_type='GP', acquisition_type='LCB', acquisition_weight=0.2,
                          eps=1e-6, batch_method='local_penalization', batch_size=1, maximize=False,
                          eval_func=cross_validate_by_year, eval_params=None, verbose=True,
                          save=True, write_to=TUNING_OUTPUT_DEFAULT, *args, **kwargs):
    """Automatically configures hyperparameters of ML algorithms. Suitable for reasonably
    small sets of params.

    Parameters
    ----------
    predictor_cls: Python class
        The predictors class.
    data: pd.DataFrame
        The data that contains the preprocessed input features and target variable.
    x_cols: list(str)
        The feature columns.
    y_col: str
        The column with the target variable.
    params: dict
        Dictionary with three keys: {"name": <str>, "type": <'discrete'/'continuous'>,
        "domain": <list/tuple>}. The continuous variables must first be specified, followed
        by the discrete ones.
    max_iter: int
        The maximum number of iterations / evaluations. Note that this excludes initial
        exploration. Also, it might converge before reaching max_iter.
    max_time: int
        The Maximum time to be used in optimization in seconds.
    model_type: str
        The model used for optimization. Defaults to Gaussian Process ('GP').
            -'GP', standard Gaussian process.
            -'GP_MCMC',  Gaussian process with prior in the hyper-parameters.
            -'sparseGP', sparse Gaussian process.
            -'warperdGP', warped Gaussian process.
            -'InputWarpedGP', input warped Gaussian process.
            -'RF', random forest (scikit-learn).
    acquisition_type: str
        Function used to determine the next parameter settings to evaluate.
            -'EI', expected improvement.
            -'EI_MCMC', integrated expected improvement (requires GP_MCMC model type).
            -'MPI', maximum probability of improvement.
            -'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model type).
            -'LCB', GP-Lower confidence bound.
            -'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model type).
    acquisition_weight: int
        Exploration vs Exploitation parameter.
    eps: float
        The minimum distance between consecutive candidates x.
    batch_method: str
        Determines the way the objective is evaluated if batch_size > 1 (all equivalent if batch_size=1).
            -'sequential', sequential evaluations.
            -'random': synchronous batch that selects the first element as in a sequential
            policy and the rest randomly.
            -'local_penalization': batch method proposed in (Gonzalez et al. 2016).
            -'thompson_sampling': batch method using Thompson sampling.
    batch_size: int
        The number of parallel optimizations to run. If None, uses batch_size = number of cores.
    verbose: bool
        Whether or not progress messages will be printed (prints if True).
    save: bool
        If set to true, will write tuning results to a pickle file at the `write_to` path.
    write_to: str
        If save=True, this defines the filepath where results are stored.
    *args, **kwargs: any
        Parameters passed to `GPyOpt.methods.BayesianOptimization` upon initialization.

    Returns
    -------
    best_params, best_score: dict, float
        the best parameters as a dictionary like {<name> -> <best value>} and the corresponding
        evaluation score.
    """
    if eval_params is None:
        eval_params = {}

    print("Using Bayesian Optimization to tune {} in {} iterations and {} seconds."
          .format(predictor_cls, max_iter, max_time))

    use_log2_scale, use_log10_scale = [], []
    for p in params:
        try:
            if p["log"] == 2:
                use_log2_scale.append(p["name"])
            elif p["log"] == 10:
                use_log10_scale.append(p["name"])
        except KeyError:
            pass

    print("Using log2-scale for parameters {} and log10-scale for {}".format(use_log2_scale, use_log10_scale))

    def create_mapping(p_arr):
        """Changes the 2d np.array from GPyOpt to a dictionary.

        Takes care of translating log-scaled parameters to their actual value
        assuming a log2 scale.

        Parameters
        ----------
        p_arr: 2d np.array
            array with parameter values in the same order as `params`.

        Returns
        -------
        mapping: dict
            Parameter mapping like {"name" -> value}.
        """
        mapping = dict()
        for i in range(len(params)):
            value = int(p_arr[0, i]) if params[i]["type"] == "discrete" else p_arr[0, i]
            if params[i]["name"] in use_log2_scale:
                value = 2**value
            elif params[i]["name"] in use_log10_scale:
                value = 10**value
            mapping[params[i]["name"]] = value

        return mapping

    def f(parameter_array):
        """The objective function to maximize."""
        param_dict = create_mapping(parameter_array)
        score = eval_func(predictor_cls, data, x_cols, y_col,
                          model_params=param_dict, **eval_params)
        results["scores"].append(score)
        results["params"].append(param_dict)
        print_eval_result(param_dict, score, verbose=verbose)
        # only return score to optimizer
        if maximize:
            return -score
        else:
            return score

    # scores are added to these lists in the optimization function f
    results = {"params": [], "scores": []}

    # run optimization in parallel
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    # set batch_size equal to num_cores if no batch_size is provided
    if not batch_size:
        batch_size = num_cores

    if verbose:
        print("Running in batches of {} on {} cores using {}."
              .format(batch_size, num_cores, batch_method))
        print("Begin training.")

    # define optimization problem
    opt = BayesianOptimization(
        f, domain=params, model_type=model_type, acquisition_type=acquisition_type,
        normalize_Y=False, acquisition_weight=acquisition_weight, num_cores=num_cores,
        batch_size=batch_size, *args, **kwargs
    )

    # run optimization
    try:
        opt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping optimization and returning results.")

    # report results
    if save:
        pickle.dump(results, open(write_to, "wb"))

    if maximize:
        best = np.argmax(results["scores"])
    else:
        best = np.argmin(results["scores"])

    best_params, best_score = results["params"][best], results["scores"][best]

    return best_params, best_score


def process_tune_results(tune_dict, ignore_params=["n_jobs"], new_score_col="AUC"):
    """Process tuning results stored in a dictionary and return them in a DataFrame.

    Parameters
    ----------
    tune_dict: dict
        Tuning results. Output of `bayesian_optimization` or stored as a pickle.
    ignore_params: list-like, default=["n_jobs"]
        Which parameters not to include in the results. Useful when a parameter was
        passed with only one possible value.
    new_score_col: str, default="AUC"
        What name to give the column referring to the evaluation/tuning score.
    """
    params = tune_dict["params"]
    param_names = [name for name in params[0].keys() if name not in ignore_params]
    param_dict = {name: [] for name in param_names}
    for observation in params:
        for name in param_names:
            param_dict[name].append(observation[name])

    param_dict[new_score_col] = tune_dict["scores"]
    df = pd.DataFrame(param_dict)
    return df


def plot_score_vs_parameter_values(data, score_col="AUC", log2_scale_params=None,
                                   log10_scale_params=None, size_by_score=False, **kwargs):
    """Plot the scores for different parameter values individually per parameter.

    Parameters
    ----------
    data: pd.DataFrame
        Output of `process_tune_results`.
    log2_scale_params: list-like, default=None
        Parameter names to scale back to log2-scale, which may improve readability of the plots.
    log10_scale_params: list-like, default=None
        Parameter names to scale back to log10-scale, which may improve readability of the plots.
    size_by_score: bool, default=False
        Whether to let dot-size depend on the score.
    **kwargs: key-value pairs
        Parameters passed to `plot_param_scatter`.

    Returns
    -------
    fig: the figure.
    """
    data = data.copy()
    # find names and number of parameters
    param_names = [col for col in data.columns if col != score_col]

    # adjust log-scale parameters
    for p in log2_scale_params:
        if p in param_names:
            data[p] = np.log2(data[p])
            data = data.rename(columns={p: "{} (log2 scale)".format(p)})

    for p in log10_scale_params:
        if p in param_names:
            data[p] = np.log10(data[p])
            data = data.rename(columns={p: "{} (log10 scale)".format(p)})

    param_names = [col for col in data.columns if col != score_col]

    return plot_param_scatter(data, score_col=score_col, size_by_score=size_by_score, **kwargs)


def plot_param_scatter(data, score_col="AUC", size_by_score=False, **kwargs):
    """Plot multiple scatterplots of the score vs each of the parameters.

    Parameters
    ----------
    data: pd.DataFrame
        Parameter values and scores, output of `process_tune_results`.
    score_col: str, default="AUC"
        Column referring to the score.
    size_by_score: bool, default=False
        Whether to let dot-size depend on the score.
    **kwargs: key-value pairs
        Parameters passed to `sns.scatterplot`.

    Returns
    -------
    fig: the figure.
    """

    def bubble_plot(x, y, size, **kwds):
        ax = sns.scatterplot(x=x, y=y, size=size, **kwds)
        ax.set_xlabel("")
        return ax

    data = data.reset_index(drop=True)
    long = data.stack().reset_index()
    long.columns = ["step", "parameter", "value"]

    scores = long.loc[long["parameter"] == score_col, ["step", "value"]]
    scores.columns = ["step", score_col]
    long = long[long["parameter"] != score_col]
    long = pd.merge(long, scores, how="left", on="step")
    long["dummy"] = long["parameter"]

    # plot
    sns.set()
    g = sns.FacetGrid(long, col="parameter", height=3.5, aspect=0.9, sharex=False, sharey=False)
    if not size_by_score:
        g.map(sns.scatterplot, "value", score_col, **kwargs)
    else:
        g.map(bubble_plot, "value", score_col, score_col, sizes=(10, 1000), alpha=.8, **kwargs)

    g.set_titles("{col_name}")
    for ax in g.axes[0]:
        ax.set_xlabel(ax.get_xlabel(), size=LABEL_SIZE)
        ax.set_ylabel(ax.get_ylabel(), size=LABEL_SIZE)
        ax.set_title(ax.get_title(), size=LABEL_SIZE)

    g.fig.suptitle("{} scores vs parameter values".format(score_col), weight="bold", size=TITLE_SIZE)
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.8)
    return g.fig


def plot_convergence(data, score_col="AUC", maximize=True):
    """Plot the convergence of the Bayesian Optimization procedure.

    Parameters
    ----------
    data: pd.DataFrame
        The tuning results, output of `process_tune_results`.
    score_col: str, default="AUC"
        Column referring to the score.
    maximize: bool, default=True
        Whether the best score is the highest (True) or lowest (False).

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        The plot showing the convergence and best score at each step of tuning.
    """
    data = data.copy()
    data = data.reset_index(drop=True)
    if maximize:
        current_best = data[score_col].min()
        data["best"] = current_best
        scores = data[score_col].values
        for i in range(len(data)):
            if scores[i] > current_best:
                current_best = scores[i]
            data["best"].iloc[i] = current_best
    else:
        current_best = data[score_col].max()
        data["best"] = current_best
        scores = data[score_col].values
        for i in range(len(data)):
            if scores[i] < current_best:
                current_best = scores[i]
            data["best"].iloc[i] = current_best

    # finalize data
    data.index.names = ["step"]
    data = data.reset_index()

    # plot
    sns.set()
    fig, ax = plt.subplots(figsize=(5, 5))

    # ax = sns.scatterplot(x="step", y="AUC", data=data, ax=ax, s=100)
    ax = sns.lineplot(x="step", y="best", data=data, ax=ax, linewidth=3, color="red")
    ax = sns.lineplot(x="step", y="AUC", data=data, ax=ax, markersize=10, markers=True, marker="o")
    ax.lines[1].set_linestyle(":")
    ax.lines[0].set_linestyle(":")
    fig.suptitle("Convergence plot", weight="bold", size=TITLE_SIZE)
    fig.subplots_adjust(top=0.92)
    ax.set_xlabel(ax.get_xlabel(), size=LABEL_SIZE)
    ax.set_ylabel(ax.get_ylabel(), size=LABEL_SIZE)
    return fig
