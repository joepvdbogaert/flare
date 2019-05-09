import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.neighbors import NearestNeighbors
from itertools import product


def load_pkl(file_path):
    return pickle.load(open(file_path, "rb"))


def to_pkl(object, file_path):
    pickle.dump(object, open(file_path, "wb"), protocol=4)


def load_grid_data(data_dir="../data/", filenames=None):
    """ Load the CBS 100m x 100m Grid data.

    Parameters
    ----------
    data_dir: str, optional
        Path to the directory that contains the grid data.
    filesnames: array-like, optional
        The names of the files / folders containing the grid shape data).
        Defaults to None, in which case it assumes the following file names:
        ["CBSvierkant100m_20002014", "CBSvierkant100m_2015", "CBSvierkant100m_2016",
        "CBSvierkant100m_2017"]

    Returns
    -------
    datasets: list
        A list containing geopandas.DataFrames for every file name in `filenames`.
    """
    if filenames is None:
        filenames = ["CBSvierkant100m_20002014", "CBSvierkant100m_2015",
                     "CBSvierkant100m_2016", "CBSvierkant100m_2017"]

    # first check if files exist to avoid errors after loading the first few files
    for filename in filenames:
        assert os.path.exists(os.path.join(data_dir, filename)), \
            ("File or directory not found. Check if you entered the correct data_dir and"
             "filenames. Path: {}".format(os.path.join(data_dir, filename)))

    datasets = []
    for filename in filenames:
        datasets.append(gpd.read_file(os.path.join(data_dir, filename)))

    return datasets


def filter_grids(datasets, min_east, max_east, min_north, max_north):
    """ Extract a square area from grids that cover The Netherlands.

    Parameters
    ----------
    datasets: array-like of geopandas.DataFrames
        The grid datasets to process.
    min_east, max_east, min_north, max_north: int
        The minimum and maximum coordinates of the resulting square. Coordinates follow
        the Dutch EPSG:28992 coordinate system.

    Returns
    -------
    datasets: array-like of geopandas.DataFrames
        The cropped grid data.
    """
    for grid in datasets:
        # add coordinates
        grid.rename(columns={"c28992r100": "C28992R100"}, inplace=True)
        grid = add_location_coordinates(
            grid,
            loc_column="C28992R100",
            x_column_name="x_coord_square_origin",
            y_column_name="y_coord_square_origin"
        )
        # filter squares
        grid = grid[(grid["x_coord_square_origin"] >= min_east) &
                    (grid["x_coord_square_origin"] < max_east) &
                    (grid["y_coord_square_origin"] >= min_north) &
                    (grid["y_coord_square_origin"] < max_north)].copy()

        # drop columns again
        grid.drop(["x_coord_square_origin", "y_coord_square_origin"], axis=1, inplace=True)

    return datasets


def add_location_coordinates(grid_data, loc_column="C28992R100", x_column_name="x_coord",
                             y_column_name="y_coord"):
    """ Extract the coordinates of the locations (origins of the squares) from the IDs. """
    grid_data[x_column_name] = grid_data["C28992R100"].str[1:5].astype(int) * 100
    grid_data[y_column_name] = grid_data["C28992R100"].str[-4:].astype(int) * 100
    return grid_data


def construct_location_ids(x_coords, y_coords):
    """Construct the 'C28992R100' code from x and y coordinates.

    Works both on a single instance as well as on vectors.

    Parameters
    ----------
    x_coords, y_coords: int, float, or array
        The x,y coordinates to convert. X and Y are assumed to be of the same length if they are
        arrays. Also, the pairs of coordinates are assumed to be in matching positions (i.e.,
        x_coords[i] belongs to y_coords[i] for all i).
    """
    x_coords = pd.Series(x_coords / 100).apply(np.floor).apply(int)
    y_coords = pd.Series(y_coords / 100).apply(np.floor).apply(int)
    return "E" + x_coords.astype(str).str.zfill(4) + "N" + y_coords.astype(str).str.zfill(4)


def split_and_rename_pre_2015_data(grid_data, min_year=2000, max_year=2014):
    """ Split the CBS grid data 2000-2014 by year and rename columns to match later years.

    In order to match the columns of data from 2015 and later, the following replacements
    are made in the column names:

    ```
    {"WON": "WONING",
     "INW": "INWONER",
     "I_": "INW_",
     "MAN_": "MAN",
     "VROUW_": "VROUW",
     "AUTO": "AUTOCHT",
     "WAL": "WALLOCH"}
    ```

    It also adds a column `YEAR` with the corresponding year to each dataset.

    Parameters
    ----------
    grid_data: geopandas.DataFrame
        The CBS grid data from 2000 to 2014.
    min_year: int, optional (default: 2000)
        The first year of which to return data.
    max_year: int, optional (default: 2014)
        The last year to return data for.

    Returns
    -------
    datasets: dict
        A dictionary like {year -> geopandas.DataFrame} where every DataFrame only contains
        columns belonging to that year and where the column names are identical to the
        corresponding column names in grid data of 2015 and later.
    """
    # replace these patterns in the column names to make them equal to data from 2015+
    col_replace = {"WON": "WONING",
                   "INW": "INWONER",
                   "I_": "INW_",
                   "MAN_": "MAN",
                   "VROUW_": "VROUW",
                   "AUTO": "AUTOCHT",
                   "WAL": "WALLOCH"}

    data_by_year = {}
    for year in np.arange(min_year, max_year+1, 1):
        # make the year a string
        year = str(year)
        # add dictionary entry with the data belong to 'year'
        data_by_year[year] = grid_data.loc[:,
            np.append("C28992R100", grid_data.columns[grid_data.columns.str.contains(year)])]
        # remove the year from the column names
        data_by_year[year].columns = data_by_year[year].columns.str.replace(year, "")
        # replace other patterns so that the column names are similar to the succeeding years (2015+)
        for key in col_replace.keys():
            data_by_year[year].columns = \
                data_by_year[year].columns.str.replace(key, col_replace[key])

        data_by_year[year]["YEAR"] = int(year)

    return data_by_year


def combine_grid_data(data2000_2014, other_datasets, other_years=[2015, 2016, 2017]):
    """ Concatenate the CBS grid data from different years.

    Parameters
    ----------
    data2000_2014: dict
        Dictionary like {'year' -> geopandas.DataFrame}, i.e., the result of
        `split_and_rename_pre_2015_data`. Every dataset must contain a column `YEAR`.
    other_datasets: array-like of geopandas.DataFrames
        Other years of CBS grid data (2015 and later).
    other_years: array-like of strings
        The years to which the datasets in `other_datasets` belong.

    Returns
    -------
    combined_data: geopandas.DataFrame
        The concatenated data.
    """
    # add the year to the other datasets
    for i in range(len(other_datasets)):
        other_datasets[i]["YEAR"] = other_years[i]

    df = pd.concat(list(data2000_2014.values()) + list(other_datasets), sort=True)

    return df


def drop_geometric_information(geo_data, geo_column="geometry"):
    """ Drop columns describing geometric attributes and return a normal data frame.

    Parameters
    ----------
    geo_data: geopandas.DataFrame
        The geometric data.
    geo_column: str or array-like of strings
        The column(s) containing the geometric information.

    Returns
    -------
    data: pandas.DataFrame
        The data without 'geo_column' as a normal (not geo-) pandas.DataFrame.
    """
    return pd.DataFrame(geo_data.drop(geo_column, axis=1))


def convert_nonnumerical_columns(data, custom_mapping=None, filna_value=0):
    """ Convert string values in the grid data to appropriate numerical values.

    Parameters
    ----------
    data: pd.DataFrame,
        The data in which to convert non-numeric values.
    custom_mapping: dict, optional,
        Custom {str -> int}. If the key is present in the default mapping, the default
        mapping is overridden, otherwise the key-value pair is added to it.
    filna_value: any, optional,
        The value with which to fill NaN values before mapping strings to integers.

    Returns
    -------
    """

    # convert to int directly, otherwise use the mapping
    def map_to_number(x):
        try:
            return int(x)
        except ValueError:
            return str_to_int_map[x]

    # define how to map string values to integers
    str_to_int_map = {
        'nihil': 0,
        'geheim': -99997,
        '-99997': -99997,
        'geen autochtoon': 0,
        'geen w. allochtoon': 0,
        'geen nw. allochtoon': 0,
        'minder dan 8%': 4,
        'minder dan 10%': 5,
        'minder dan 40%': 2,
        '8% tot 15%': 12,
        '10% tot 25%': 18,
        '15% tot 25%': 20,
        '25% tot 45%': 35,
        '40% tot 60%': 50,
        '45% tot 67%': 55,
        '60% tot 75%': 68,
        '75% tot 90%': 83,
        '45% of meer': 78,
        '67% of meer': 75,
        '90% of meer': 95,
    }

    # override or add custom mapping entries
    if custom_mapping is not None:
        for key, value in custom_mapping.items():
            str_to_int_map[key] = value

    # get all non-numerical columns
    non_numericals = data.loc[:, data.dtypes == 'object'].copy()
    if "C28992R100" in non_numericals.columns:
        non_numericals.drop("C28992R100", axis=1, inplace=True)

    # convert to int directly or map according to the mapping
    converted = non_numericals.fillna(filna_value).applymap(lambda x: map_to_number(x))

    # convert data type to integer
    for col in converted.columns:
        converted[col] = converted[col].astype(int)

    # put the new columns back in the dataframe and return
    data[converted.columns] = converted.values
    return data


def set_negative_to_value(data, value, skip_columns=["C28992R100"]):
    """ Set all negative values in a DataFrame to a single value.

    Parameters
    ----------
    data: pd.DataFrame,
        The data to adjust.
    value: int, float, str
        The value to set all negative values to.

    Returns
    -------
    data: pd.DataFrame,
        The data where every negative entry in the original data has been
        replaced with `value`.
    """
    # loop for memory efficiency
    for col in data.columns:
        if col not in skip_columns:
            data[col] = np.where(data[col] < 0, value, data[col])

    return data


def replace_shuffled_columns(data, new_columns, loc_column="C28992R100", time_column="YEAR"):
    """ Put the new columns in the data when the rows have been shuffled.

    Parameters
    ----------
    data: pd.DataFrame,
        The full dataset.
    new_columns: pd.DataFrame,
        The data that should be inserted in the full dataset, where the rows are possibly
        shuffled.
    loc_column: str, optional (default: "C28992R100"),
        The column indicating the location (square) of each record.
    time_column: str, optional (default: "YEAR"),
        The column indicating the time of each record.

    Returns
    -------
    data: pd.DataFrame,
        The data where the data in new_columns replaces the original data in the same columns.
    """
    data.sort_values([loc_column, time_column], inplace=True)
    new_columns.sort_values([loc_column, time_column], inplace=True)

    print("Checking if order of datasets matches..")
    assert np.all(data[loc_column].values == new_columns[loc_column].values), \
        "Location column not sorted identically"
    assert np.all(data[time_column].values == new_columns[time_column].values), \
        "Time column not sorted identically"

    data[new_columns.columns] = new_columns
    return data


def get_facility_columns(data):
    return [col for col in data.columns if (col[0:2] == 'AV') or (col[0:2] == 'AF')]


def get_inhabitant_columns(data):
    return [col for col in data.columns if ("INW" in col) or
            ("OCH" in col) or col in ["MAN", "VROUW", "UITKMINAOW", "GEBOORTE"]]


def get_housing_columns(data):
    return [col for col in data.columns if ("WON" in col and not "WONER" in col)
            or ("HH" in col)]


def forward_and_backward_fill(data, sort_by=None):
    return data.sort_values(sort_by).fillna(method="ffill").fillna(method="bfill")


def fill_missing_facility_values(data, loc_column="C28992R100", time_column="YEAR", **kwargs):
    """ Fill the missing and secret values in the facilities columns.

    Performs the following steps:
        1. convert confidential values to NaNs (confidential values do not imply anything
            about the facilities)
        2. forward and backward fill per location over time
        3. use k-nearest neighbors to fill in remaining missing values

    Parameters
    ----------
    data: pd.DataFrame
        The data in which to fill facility-related columns.
    loc_column: str, optional (default: "C28992R100"),
        The name of the column that identifies the locations.
    time_column: str, optional (default: "YEAR"),
        The name of the column that represents time. This method is designed for use with
        a time column specifying the year of the data, since CBS produces separate datasets
        for each year (starting from 2015). Using smaller time units may lead to unexpected
        behavior (e.g., long computation times and memory issues).
    **kwargs: any,
        Arguments passed to `fill_missing_values_with_knn()`.

    Returns
    -------
    data: pd.DataFrame
        The same as the input data, but with housing-related columns filled.

    Notes
    -----
    Uses the `get_facility_columns()` function to obtain all columns related to facilities.
    """
    facility_columns = get_facility_columns(data)
    facility_data = data[[loc_column, time_column] + facility_columns]
    facility_data = add_location_coordinates(facility_data, x_column_name="x_coord", y_column_name="y_coord")

    # 1. convert confidential values to NaN as they do not imply anything about facilities
    facility_data = confidential_to_value(facility_data, [-99997, -99997.0], value=np.nan)

    # 2. Perform kNN on the 2015 entries, since these are the only ones available at all
    print("Imputing missing values with kNN...")
    dist_cols = [col for col in facility_columns if 'AF' in col]
    mean_cols = [col for col in facility_columns if 'AV' in col]

    # temporarily split the data to make the algorithm more efficient
    old_shape = facility_data.shape
    df2015 = facility_data[facility_data[time_column] == 2015].copy()
    df_rest = facility_data[facility_data[time_column] != 2015].copy()

    df2015 = fill_missing_values_with_knn(
        df2015,
        dist_cols,
        fill_method="min_distance_plus_value",
        **kwargs
    )

    df2015 = fill_missing_values_with_knn(
        df2015,
        mean_cols,
        fill_method="mean",
        **kwargs
    )

    # put together again
    merged_data = pd.concat([df_rest, df2015], axis=0, ignore_index=True)
    assert merged_data.shape == old_shape, \
        "Old and new shapes do not match. Old: {}. New: {}".format(old_shape, merged_data.shape)

    # 3. forward and backward fill
    filled_data = forward_backward_fill_per_location(merged_data,
                                                     loc_column=loc_column,
                                                     sort_column=time_column)

    # put filled data back in the original data
    return replace_shuffled_columns(data, filled_data, loc_column=loc_column,
                                    time_column=time_column)


def fill_missing_housing_values(data, loc_column="C28992R100", time_column="YEAR"):
    """ Fill the missing and secret values in the facilities columns.

    Parameters
    ----------
    data: pd.DataFrame
        The data in which to fill housing-related columns.
    loc_column: str, optional (default: "C28992R100"),
        The name of the column that identifies the locations.
    time_column: str, optional (default: "YEAR"),
        The name of the column that represents time. This method is designed for use with
        a time column specifying the year of the data, since CBS produces separate datasets
        for each year (starting from 2015). Using smaller time units may lead to unexpected
        behavior (e.g., long computation times and memory issues).

    Returns
    -------
    data: pd.DataFrame
        The same as the input data, but with housing-related columns filled.

    Notes
    -----
    Uses the `get_housing_columns()` function to obtain all columns related to housing and
    households.
    """
    print("Filling missing values in housing data.")
    housing_columns = get_housing_columns(data)
    mean_columns = ["GEM_HH_GR", "WOZWONING", "G_ELEK_WON", "G_GAS_WON"]
    other_cols = [col for col in housing_columns if col not in mean_columns]
    housing_data = data[[loc_column, time_column] + housing_columns]

    # 1. confidential values to NaNs for specific columns
    housing_data = confidential_to_value(housing_data, [-99997, -99997.0], value=np.nan,
                                         columns=mean_columns)

    # 2. confidential values to zero for the others
    housing_data = confidential_to_value(housing_data, [-99997, -99997.0], value=0,
                                         columns=other_cols)

    # 3. forward and backward fill per location
    housing_data = forward_backward_fill_per_location(housing_data)

    # 4. fill specific columns with median values
    for col in mean_columns:
        median_of_col = housing_data[col].median()
        housing_data[col] = housing_data[col].fillna(median_of_col)

    # 5. fill missing with zero for the rest
    housing_data.fillna(0, inplace=True)

    return replace_shuffled_columns(data, housing_data, loc_column=loc_column,
                                    time_column=time_column)


def fill_missing_inhabitant_values(data, loc_column="C28992R100", time_column="YEAR"):
    """ Fill the missing and secret values in the inhabitant columns.

    Parameters
    ----------
    data: pd.DataFrame
        The data in which to fill inhabitant-related columns.
    loc_column: str, optional (default: "C28992R100"),
        The name of the column that identifies the locations.
    time_column: str, optional (default: "YEAR"),
        The name of the column that represents time. This method is designed for use with
        a time column specifying the year of the data, since CBS produces separate datasets
        for each year (starting from 2015). Using smaller time units may lead to unexpected
        behavior (e.g., long computation times and memory issues).

    Returns
    -------
    data: pd.DataFrame
        The same as the input data, but with housing-related columns filled.

    Notes
    -----
    Uses the `get_inhabitant_columns()` function to obtain all columns related to the number
    and of inhabitants and their division over several subgroups.
    """
    print("Filling missing values in inhabitant data.")
    inhabitant_columns = get_inhabitant_columns(data)
    inhabitant_data = data[[loc_column, time_column] + inhabitant_columns]

    # 1. set confidential values to zero
    inhabitant_data = confidential_to_value(inhabitant_data, [-99997, -99997.0], value=np.nan)

    # 2. forward and backward fill
    inhabitant_data = forward_backward_fill_per_location(inhabitant_data)

    # 3. set remaining missing values to zero
    inhabitant_data.fillna(0, inplace=True)

    return replace_shuffled_columns(data, inhabitant_data, loc_column=loc_column,
                                    time_column=time_column)


def confidential_to_value(data, secret_values, value=np.nan, columns=None):
    """ Replace values indicating the real value is confidential with another value. """
    if columns is not None:
        data.loc[:, columns] = data.loc[:, columns].replace(to_replace=secret_values, value=value)
        return data
    else:
        return data.replace(to_replace=secret_values, value=np.nan)


def forward_backward_fill_per_location(data, loc_column="C28992R100", sort_column="YEAR"):
    """ Perform forward and backward fill (in that order) per location over the time periods. """
    print("Forward and backward filling per location..")
    filled_data = (data.groupby(loc_column)
                       .apply(lambda x: forward_and_backward_fill(x, sort_by=sort_column))
                       .reset_index(drop=True))
    return filled_data


def split_data_for_knn(data, y_col, features):
    """ Split the data based on missingness in a column.

    Parameters
    ----------
    data: pd.DataFrame,
        The data to split.
    y_col: str,
        The target column.
    features: array-like or str
        The columns to use as features in x_train and x_test.

    Returns
    -------
    x_train: pd.DataFrame,
        A slice of the dataframe where the columns are in `features` and the records are those
        where y_col is not missing.
    y_train: pd.DataFrame,
        The values in y_col corresponding to the records in x_train.
    x_test: pd.DataFrame,
        A slice of the dataframe where the columns are in `features` and the records are those
        where y_col is missing.
    """
    x_all = data.loc[:, features].copy()
    y_all = data.loc[:, y_col].copy()

    y_train = y_all[~y_all.isnull()].copy()
    x_train = x_all[~y_all.isnull()].copy()
    x_test = x_all[y_all.isnull()].copy()

    return x_train, y_train, x_test


def get_k_nearest_neighbor_values(x_train, y_train, x_test, K=8, n_jobs=-1):
    """ Get the K nearest neighbor values.

    Parameters
    ----------
    x_train: pd.DataFrame,
        The train features.
    y_train: pd.Series,
        The train labels / target values.
    x_test: pd.DataFrame,
        The data to predict.
    K: int, optional (default: 8)
        The number of neigbors to find.
    n_job: int, optional (default: -1)
        The number of CPU cores to use in the kNN algorithm. A value of -1 uses all available cores.

    Returns
    -------
    distances: np.array
        The distances to the K neighbors for every record in x_test. Shape: (len(x_test), K).
    predictions: np.array
        The values of the target of the K neighbors for every record in x_test. Shape (len(x_test), K).
    """
    # fit the data
    knn = NearestNeighbors(n_neighbors=K, n_jobs=n_jobs)
    knn.fit(x_train)
    # find nearest neighbors
    distances, indices = knn.kneighbors(x_test, return_distance=True)
    # get the predictions of the nearest neighbors
    predictions = np.concatenate(
        [y_train.values[indices[:, k]].reshape(-1, 1) for k in range(K)], axis=1)

    return distances, predictions


def get_minimum_total_distance_to_facility(predictions, distances, distance_factor=1.3*1e-3,
                                           *args, **kwargs):
    """ Based on nearest neighbor values and distances, retrieve the distance to the closest facility. """
    return np.min(distances * distance_factor + predictions, axis=1)


def get_mean_value_among_neighbors(predictions, *args, **kwargs):
    """ Calculate mean value of neighbors. """
    return np.mean(predictions, axis=1)


def fill_missing_values_with_knn(data, columns_to_fill, K=8, coordinate_cols=["x_coord", "y_coord"],
                                 fill_method="min_distance_plus_value", distance_factor=1.3*1e-3,
                                 index_col="C28992R100"):
    """ Fill the missing values of columns using the k-Neaest Neighbor algorithm.

    Parameters
    ----------
    data: pd.DataFrame,
        The data in which columns should be filled.
    columns_to_fill: array-like,
        The columns in the data to fill using kNN.
    K: int, optional (default: 8)
        The number of neighbors to use in the kNN algorithm.
    coordinate_cols: array-like, optional (default: ['x_coord', 'y_coord'])
        The columns to use in calculating the distance. Intended use is based on two columns
        that define the x and y coordinates in ESRI 28992, so that the L2 distance equals the
        distance 'as the crow flies' between two points.
    fill_method: str, one of ['distance_plus_value', 'mean'], optional,
        If 'min_distance_plus_value', the fill values are calculated as the minimum of
        `distance * distance_factor + value_of_neighbor` over the K neighbors. If 'mean', the
        fill values are the simple average of the values of the K neighbors.
    distance_factor: float,
        The value to multiply the distances with in case `fill_method='min_distance_plus_value'
        so that the distance and value correspond in unit. E.g., if coordinates in meters are
        used to calculate the distance, but the value column is in kilometers, you can specify
        `distance_factor = 1e-3` to compensate. In addition, to account for the fact that the
        distance is 'as the crow flies', a factor somewhat bigger than 1 could be used to estimate
        a distance over the road. This combined, leads to the default value of `1.3*1e-3`. Note
        that this argument is ignored when the fill_method is not 'min_distance_plus_value'.

    Returns
    -------
    data: pd.DataFrame,
        The filled data.
    """
    def has_missing(series):
        return (series.isnull().sum() > 0)

    if fill_method == "min_distance_plus_value":
        func = get_minimum_total_distance_to_facility
    elif fill_method == "mean":
        func = get_mean_value_among_neighbors

    if isinstance(columns_to_fill, str):
        columns_to_fill = [columns_to_fill]

    data = data.set_index(index_col)
    for target_column in columns_to_fill:
        if has_missing(data[target_column]):
            print("\rFilling column: {}".format(target_column), end="")
            x_train, y_train, x_test = split_data_for_knn(data, target_column, coordinate_cols)
            # get nearest neighbors
            distances, predictions = get_k_nearest_neighbor_values(x_train, y_train, x_test, K=K)
            # get aggregated value according to 'fill_method'
            values = func(predictions, distances, distance_factor=distance_factor)
            # impute the found values
            data.loc[x_test.index, target_column] = values
        else:
            print("\rSkipping {}, because there are no missing values.".format(target_column))
    print("\rColumns filled")

    return data.reset_index(drop=False)


def impute_missing_values(data, loc_column="C28992R100", time_column="YEAR"):
    """ Impute mergedissing values in the merged CBS Grid data. """
    data = insert_missing_time_location_combinations(data, loc_column=loc_column, time_column=time_column)
    data = fill_missing_facility_values(data, loc_column=loc_column, time_column=time_column)
    data = fill_missing_housing_values(data, loc_column=loc_column, time_column=time_column)
    data = fill_missing_inhabitant_values(data, loc_column=loc_column, time_column=time_column)
    return data


def insert_missing_time_location_combinations(data, loc_column="C28992R100", time_column="YEAR"):
    """ Add missing combinations of time and location by reindexing on their uniqueu values. """
    locations = data[loc_column].unique()
    years = data[time_column].unique()
    old_shape = data.shape
    data.set_index([loc_column, time_column], inplace=True)
    data = data.reindex([x for x in product(locations, years)], fill_value=np.nan, copy=False).reset_index()
    print("Dataset size increased from {} to {} rows.".format(old_shape[0], data.shape[0]))
    return data


def filter_no_location_incidents(incidents, coord_cols=["st_x", "st_y"]):
    """Filter out incidents that have no usable location information.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data.
    coord_cols: list of length 2, default=["st_x", "st_y"]
        The columns representing the coordinates.
    """
    mask = (incidents[coord_cols[0]] == 0) | (incidents[coord_cols[1]] == 0)
    num = np.sum(mask)
    length = len(incidents)
    data = incidents[~mask]
    print("Removed {} incidents ({:.3f}%) from data".format(num, num/length*100))
    return data


def add_grid_id_column(incidents, x_col="st_x", y_col="st_y", new_col="C28992R100"):
    """Add a column to the incident data with the 'C28992R100' grid ID.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data.
    x_col, y_col: str, default='st_x' and 'st_y'
        The column names of the x and y coordinates.
    """
    incidents.loc[:, new_col] = construct_location_ids(incidents["st_x"], incidents["st_y"])
    return incidents


def merge_grid_with_incidents(incidents, grid, square_col="C28992R100", inc_year_col="dim_datum_jaar",
                              grid_year_col="YEAR", type_col="dim_incident_incident_type",
                              x_coord_col="st_x", y_coord_col="st_y", verbose=False):
    """Map incidents to squares of the CBS grid and merge the data accordingly.

    Some assumptions are made based on the flare.preprocessing pipeline of the grid data:
        - the grid data is filtered to only contain years for which we have all the data
        - the grid contains all the possible squares in the region (plus possibly more)

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data.
    grid: pd.DataFrame
        The grid data, preprocessed and containing all relevant years of CBS statistics
        as well as all relevant squares.
    square_col: str, default="C28992R100"
        The column with square IDs.
    inc_year_col: str, default="dim_datum_jaar"
        The column in the incident data with the year.
    grid_year_col: str, default="YEAR"
        The column in the grid data with the year.
    type_col: str, default="dim_incident_incident_type"
        The column in the incident data with the incident type.

    Returns
    -------
    merged_df: pd.DataFrame
        A DataFrame with both the CBS statistics as well as the number of incidents
        of each type that happened for each square-year combination.
    """
    # drop incidents with no usable location information
    incidents = filter_no_location_incidents(incidents, coord_cols=[x_coord_col, y_coord_col])
    # determine the square ID of each incident based on coordinates
    incidents = add_grid_id_column(incidents, x_col=x_coord_col, y_col=y_coord_col, new_col=square_col)
    # count incidents of each type per square
    grouped = (incidents.groupby([square_col, inc_year_col, type_col])
                        [incidents.columns[0]]
                        .count())

    # reindex
    if verbose:
        print("Reindexing to obtain all combinations..")

    new_index = pd.MultiIndex.from_tuples(
        [tup for tup in product(incidents[square_col].unique(), incidents[inc_year_col].unique(), incidents[type_col].unique())],
        names=[square_col, inc_year_col, type_col]
    )
    grouped = grouped.reindex(new_index, fill_value=0)

    grouped.index.droplevel(type_col)
    transformed = (grouped.unstack(fill_value=0)
                          .rename_axis(None, axis=1)
                          .reset_index())

    # and merge it with the grid data
    if verbose:
        print("Transformed incident data has shape: {}".format(transformed.shape))
        print("grid data has shape: {}".format(grid.shape))
    merged = transformed.merge(grid, left_on=[square_col, inc_year_col],
                               right_on=[square_col, grid_year_col], how="left")

    if verbose:
        print("merged data has shape: {}".format(merged.shape))
        print("dropping incomplete years (assuming this was done in grid data already)")
    minyear, maxyear = grid[grid_year_col].min(), grid[grid_year_col].max()
    merged = merged[(merged[inc_year_col] >= minyear) & (merged[inc_year_col] <= maxyear)]

    if verbose:
        print("new shape: {}".format(merged.shape))
        print("dropping irrelevant squares")
    merged.dropna(subset=list(grid.columns), inplace=True)

    if verbose:
        print("final shape of merged data: {}".format(merged.shape))
    return merged
