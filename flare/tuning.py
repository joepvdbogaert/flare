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
                          save=True, write_to=TUNING_OUTPUT_DEFAULT):
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

    use_log_scale = []
    for p in params:
        try:
            if p["log"]:
                use_log_scale.append(p["name"])
        except KeyError:
            pass

    print("Using log2-scale for parameters {}".format(use_log_scale))

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
            if params[i]["name"] in use_log_scale:
                value = 2**value
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
        batch_size=batch_size
    )

    # run optimization
    try:
        opt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping optimization and returning results.")

    # report results
    if save:
        pickle.dump(results, open(write_to, "wb"))

    best = np.argmax(results["scores"])
    best_params, best_score = results["params"][best], results["scores"][best]

    return best_params, best_score
