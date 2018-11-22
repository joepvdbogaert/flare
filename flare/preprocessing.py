import os
import numpy as np
import pandas as pd
import geopandas as gpd


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
        grid.rename(columns={"c28992r100": "C28992R100"}, inplace=True)
        grid["x_coord_square_origin"] = grid["C28992R100"].str[1:5].astype(int) * 100
        grid["y_coord_square_origin"] = grid["C28992R100"].str[-4:].astype(int) * 100
        grid = grid[(grid["x_coord_square_origin"] >= min_east) &
                    (grid["x_coord_square_origin"] < max_east) &
                    (grid["y_coord_square_origin"] >= min_north) &
                    (grid["y_coord_square_origin"] < max_north)].copy()

        grid.drop(["x_coord_square_origin", "y_coord_square_origin"], axis=1, inplace=True)

    return datasets


def split_and_rename_pre_2015_data(grid_data):
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
    for year in np.arange(2000, 2015, 1):
        # make the year a string
        year = str(year)
        # add dictionary entry with the data belong to 'year'
        data_by_year[year] = grid_data.loc[:,np.append("C28992R100", grid_data.columns[grid_data.columns.str.contains(year)])]
        # remove the year from the column names
        data_by_year[year].columns = data_by_year[year].columns.str.replace(year, "")
        # replace other patterns so that the column names are similar to the succeeding years (2015+)
        for key in col_replace.keys():
            data_by_year[year].columns = data_by_year[year].columns.str.replace(key, col_replace[key])

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
