import numpy as np
import pandas as pd
import seaborn as sns

from flare.preprocessing import (
    get_housing_columns,
    get_inhabitant_columns,
    get_facility_columns
)


def tanh_scaling(x, alpha=1.0):
    """Apply tanh with min-max scaling.

    Applies the following formula:

    ..math::

        x_{scaled} = \text{tanh} \big[ 2 * \frac{x - x_{min}}{x_{max} - x_{min}} -1 \big]


     Parameters
    ----------
    x: np.array
        The data to scale.
    alpha: float
        Scaling factor that is applied to the min-max scaled value before applying tanh.

    Returns
    -------
    x_scaled: np.array
        The scaled data.
    """
    return np.tanh(alpha * 2 * (x - x.min()) / (x.max() - x.min()) - 1)


def log_scaling(x, constant=1e-2, divide=5):
    """Apply log-scaling.

    A constant is added to the entire array to prevent zero entries. This assumes the data
    itself is non-negative.

    ..math::

        x_{scaled} = \frac{\text{log}(x + \text{constant})}{\text{divide}}

    Parameters
    ----------
    x: np.array
        The data to scale.
    constant: float
        The constant value to add to the array to avoid :math:`log(0)`.
    divide: int, float, default=5
        Additional factor to divide the result by to further scale down/up the result.

    Returns
    -------
    x_scaled: np.array
        The scaled data.
    """
    return np.log(x + constant) / divide


def safe_divide(num, denom, fill_value=0):
    """Safely divide one array by another element-wise.

    Entries where the denominator is zero are filled by a custom value.

    Parameters
    ----------
    num, denom: np.array
        The numerator and denominator arrays.
    fill_value: float, int, string, default=0
        The value to insert where denominator is zero.

    Returns
    -------
    result: np.array
        The divided array with zeros where denominator is zero.
    """
    return np.where(denom == 0, fill_value, num / denom)


def plot_features_distributions(data, scale_func=None, xlim=None, ylim=None, bins=None, ignore=None):
    """Plot histograms of all features in a dataset.

    Parameters
    ----------
    data: pd.DataFrame
        The data to plot.
    scale_func: function, default=None
        The function to use to scale the data before plotting.
    xlim, ylim: tuple(int, int)
        The limits of the plot axes.
    bins: array-like
        The bin split-values for the histogram.
    ignore: array-like
        Any column names not to plot.
    """
    if ignore is not None:
        data = data.drop(ignore, axis=1)
    # to long format
    long_data = data.stack(0).reset_index(1).rename(columns={'level_1': 'feature', 0: 'value'})
    # plot facetgrid
    if scale_func is None:
        g = sns.FacetGrid(long_data, col="feature", col_wrap=5, xlim=xlim, ylim=ylim)
        g.map(sns.distplot, 'value', bins=bins, norm_hist=True)
    else:
        long_data["scaled value"] = scale_func(long_data['value'])
        g = sns.FacetGrid(long_data, col="feature", col_wrap=5, xlim=xlim)
        g.map(sns.distplot, 'scaled value', bins=bins, norm_hist=True)

    plt.show()


def select_and_scale_facility_cols(data, scale_func=log_scaling):
    """Drop irrelevant facility columns and scale the others.

    Some columns are selected by this function and others are dropped. Specifically, we
    keep the columns corresponding to the number of facilities within 1, 3, and 5 kilometers
    over the road. Longer ranges and the distances to the closest facilities are dropped, since
    they are not informative.

    All the kept facility columns are scaled using log_scaling by default, but this can be
    modified.

    Parameters
    ----------
    data: pd.DataFrame
        The CBS data with facility features.
    scale_func: function, default=`flare.scaling.log_scaling`.
        The function to use to scale the data before plotting.

    Returns
    -------
    scaled_data: pd.DataFrame
        The data where the facility columns are scaled and some columns are dropped.
    
    Notes
    -----
    The data cannot contain any missing values. The facility columns are retrieved using
    `flare.preprocessing.get_facility_columns`.
    """
    scaled = data.copy()

    # organize columns and select those to keep
    facility_cols = get_facility_columns(data)
    av1_cols = [col for col in facility_cols if "AV1_" in col]
    av3_cols = [col for col in facility_cols if "AV3_" in col]
    av5_cols = [col for col in facility_cols if "AV5_" in col]
    keep = av1_cols + av3_cols + av5_cols

    # drop other columns
    to_drop = list(set(facility_cols) - set(keep))
    scaled = scaled.drop(to_drop, axis=1)

    # scale
    scaled[keep] = scaled[keep].apply(lambda x: scale_func(x))

    return scaled


def scale_inhabitant_cols(data, scale_func=None):
    """Scales the inhabitant columns of the CBS data, so that they are dimensionless and
    close to zero.

    The columns are obtained using `flare.preprocessing.get_inhabitant_columns`. All of these
    columns are assumed to be in the data.

    Parameters
    ----------
    data: pd.DataFrame
        The data with relevant columns to scale.
    scale_func: function, default=`flare.scaling.log_scaling`.
        Used to scale the 'INWONER' (number of residents) column.

    Returns
    -------
    scaled_data: pd.DataFrame
        The data where the inhabitant columns are scaled.
    """
    scaled = data.copy()
    inhabitant_cols = get_inhabitant_columns(data)
    perc_cols = [col for col in inhabitant_cols if col[:2] == "P_"]
    divide_by_residents_cols = [col for col in inhabitant_cols if
                                ("INW_" in col) or (col in ["MAN", "VROUW", "GEBOORTE", "UITKMINAOW"])]

    assert np.all(np.in1d(perc_cols, data.columns))
    assert np.all(np.in1d(divide_by_residents_cols, data.columns))
    assert "INWONER" in data.columns

    scaled[perc_cols] = scaled[perc_cols] / 100
    scaled[divide_by_residents_cols] = scaled[divide_by_residents_cols].apply(lambda x: safe_divide(x, scaled["INWONER"].values))

    # correct if MAN + VROUW > 1 or UITMINAOW > 1
    scaled["UITKMINAOW"] = np.minimum(1., scaled["UITKMINAOW"])
    scaled[["MAN", "VROUW"]] = scaled[["MAN", "VROUW"]].apply(lambda x: safe_divide(x, scaled["MAN"] + scaled["VROUW"]))
    
    INW_cols = [col for col in inhabitant_cols if "INW_" in col]
    scaled[INW_cols] = scaled[INW_cols].apply(lambda x: safe_divide(x, scaled[INW_cols].sum(axis=1)))
    
    if scale_func is not None:
        scaled["INWONER"] = scale_func(scaled["INWONER"].values)

    return scaled


def scale_housing_cols(data, scale_func=log_scaling):
    """Scales the inhabitant columns of the CBS data, so that they are dimensionless and
    close to zero.

    The columns are obtained using `flare.preprocessing.get_housing_cols`. All of these
    columns are assumed to be in the data. The columns are further split into different
    subsets to apply different scaling:

    - ["G_ELEK_WON", "G_GAS_WON"]: scaled by mean :math:`(x - x_{mean}) / x_{mean}).
    - columns starting with "P_": divided by 100 to change percentage to proportion.
    - ["WON_HCORP", "WON_MRGEZ", "WON_LEEGST", "WONVOOR45"]: divide by the number of houses
      to obtain a proportion.
    - columns specifying the number of houses built in given periods: divided by the number
      of houses and then divided by their sum so that they sum to 1.
    - columns specifying the numbers of different types of households: divided by the total
      number of households to obtain a proportion and divided by their sum so that they sum
      to 1.
    - ["WONING", "AANTAL_HH", "WOZWONING"]: custom scale function (log-scaling by default).

    Parameters
    ----------
    data: pd.DataFrame
        The data with relevant columns to scale.
    scale_func: function, default=`flare.scaling.log_scaling`.
        Used to scale the 'INWONER' (number of residents) column.

    Returns
    -------
    scaled_data: pd.DataFrame
        The data where the housing columns are scaled.
    """
    housing_cols = get_housing_columns(data)
    mean_scaling_cols = ["G_ELEK_WON", "G_GAS_WON"]
    perc_cols = [col for col in housing_cols if col[:2] == "P_"]
    divide_by_houses = ["WON_HCORP", "WON_MRGEZ", "WON_LEEGST", "WONVOOR45"]
    built_years = [col for col in housing_cols if (col[:4] == "WON_") and (col not in divide_by_houses)]
    divide_by_households = [col for col in housing_cols if "HH_" in col]    
    custom_scale_cols = ["WONING", "AANTAL_HH", "WOZWONING"]

    assert np.all(np.in1d(housing_cols, mean_scaling_cols + perc_cols + divide_by_houses + divide_by_households + custom_scale_cols + built_years)), "missing: {}".format(
        set(housing_cols) - set(mean_scaling_cols + perc_cols + divide_by_houses_cols + divide_by_households + custom_scale_cols + built_years))

    scaled = data.copy()
    scaled[mean_scaling_cols] = scaled[mean_scaling_cols].apply(lambda x: (x - x.mean()) / x.mean())
    scaled[perc_cols] = scaled[perc_cols] / 100
    scaled[divide_by_houses] = scaled[divide_by_houses].apply(lambda x: np.minimum(1., safe_divide(x.values, scaled["WONING"].values)))
    scaled[divide_by_households] = scaled[divide_by_households].apply(lambda x: safe_divide(x.values, scaled["AANTAL_HH"].values))
    scaled[built_years] = scaled[built_years].apply(lambda x: safe_divide(x, scaled[built_years].sum(axis=1).values))

    if scale_func is not None:
        scaled[custom_scale_cols] = scaled[custom_scale_cols].apply(lambda x: scale_func(x))

    return scaled


def scale_demographics_features(data, scale_func=log_scaling):
    """Scale CBS demographics data so that they can be used in clustering and prediction
    models.

    Parameters
    ----------
    data: pd.DataFrame
        The data that contains CBS demographics data.
    scale_func: Python function, default=log_scaling
        The function that is used to scale several absolute numbers.

    Returns
    -------
    data_scaled: pd.DataFrame
        The data where all CBS demographics features are either scaled or dropped.

    Notes
    -----
    More details on which scaling is applied to which columns and which columns are dropped
    can be found in the three called functions: `flare.scaling.scale_housing_cols`,
    `flare.scaling.scale_inhabitant_cols`, and `flare.scaling.select_and_scale_facility_cols`.
    """
    df_scaled = scale_housing_cols(data, scale_func=log_scaling)
    df_scaled = scale_inhabitant_cols(df_scaled, scale_func=log_scaling)
    df_scaled = select_and_scale_facility_cols(df_scaled, scale_func=log_scaling)
    return df_scaled
