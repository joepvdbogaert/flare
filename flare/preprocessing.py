import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd


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


def construct_location_id(x_coords, y_coords):
    return "E" + np.array(x_coords/100, dtype=str) + "N" + np.array(y_coords/100, dtype=str)


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
    return [col for col in data.columns if ("WON_" in col and not "WONER" in col)
            or ("HH" in col)]


def forward_and_backward_fill(data, sort_by=None):
    return data.sort_values(sort_by).fillna(method="ffill").fillna(method="bfill")


def fill_missing_facility_values(data, loc_column="C28992R100", time_column="YEAR"):
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

    # 1. convert confidential values to NaN as they do not imply anything about facilities
    facility_data = confidential_to_value(facility_data, [-99997, -99997.0], value=np.nan)

    # 2. forward and backward fill
    filled_data = forward_backward_fill_per_location(facility_data,
                                                     loc_column=loc_column,
                                                     sort_column=time_column)

    # 3. fill remaining with nearest neighbor approach based on distance of locations
    # todo..

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
    housing_data = data[[loc_column, time_column] + housing_columns]

    print("Replacing confidential values...")
    # 1. confidential values to NaNs for specific columns
    housing_data = confidential_to_value(housing_data, [-99997, -99997.0], value=np.nan,
                                         columns=mean_columns)

    # 2. confidential values to zero for the others
    other_cols = list(set(housing_data.columns) - set(mean_columns))
    housing_data = confidential_to_value(housing_data, [-99997, -99997.0], value=np.nan,
                                         columns=other_cols)

    # 3. forward and backward fill per location
    housing_data = forward_backward_fill_per_location(housing_data)

    print("Filling the remaining values with zeros or median values.")
    # 4. fill specific columns with median values
    for col in mean_columns:
        median_of_col = housing_data[col].median()
        housing_data[col] = housing_data.fillna(median_of_col)

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
    confidential_to_value(inhabitant_data, [-99997, -99997.0], value=np.nan)

    # 2. forward and backward fill
    inhabitant_data = forward_backward_fill_per_location(inhabitant_data)

    # 3. set remaining missing values to zero
    inhabitant_data.fillna(0, inplace=True)

    return replace_shuffled_columns(data, inhabitant_data, loc_column=loc_column,
                                    time_column=time_column)


def confidential_to_value(data, secret_values, value=np.nan, columns=None):
    """ Replace values indicating the real value is confidential with another value. """
    if columns is not None:
        data[columns] = data[columns].replace(to_replace=secret_values, value=value)
        return data
    else:
        return data.replace(to_replace=secret_values, value=np.nan)


def forward_backward_fill_per_location(data, loc_column="C28992R100", sort_column="YEAR"):
    """ Perform forward and backward fill (in that order) per location over the time periods. """
    print("Grouping by location and filling..")
    filled_data = (data.groupby(loc_column)
                       .apply(lambda x: forward_and_backward_fill(x, sort_by=sort_column))
                       .reset_index(drop=True))
    return filled_data
