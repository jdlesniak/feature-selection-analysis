import pandas as pd
import numpy as np
from scipy.io import arff
import pprint

import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

### get our util functions and paths
from utils import *
from paths import RAW_DATA_DIR, SEISMIC_DATA, CLEAN_DATA_DIR, SEED

def describe_column(df, col):
    """
    Analyze the data type and summarize values for a given column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the column to describe.
        col (str): The name of the column to describe.

    Returns:
        dict: A dictionary containing the column name as key and a nested dictionary of:
            - dtype (str): Data type of the column
            - For numeric columns:
                - min (float): Minimum value
                - max (float): Maximum value
                - mean (float): Mean value
                - std (float): Standard deviation
                - n_unique (int): Number of unique values
            - For categorical columns:
                - n_unique (int): Number of unique categories
                - most_common (object): Most frequent category
                - most_common_freq (int): Frequency of the most common category
                - unique_values (list): List of all unique values
    """
    col_data = df[col]
    dtype = col_data.dtype

    if isinstance(col_data.iloc[0], (int, float)):
        desc = {
            "dtype": str(dtype),
            "min": col_data.min(),
            "max": col_data.max(),
            "mean": col_data.mean(),
            "std": col_data.std(),
            "n_unique": col_data.nunique()
        }
    elif isinstance(col_data.iloc[0], (bytes, str)):
        desc = {
            "dtype": str(dtype),
            "n_unique": col_data.nunique(),
            "most_common": col_data.value_counts().idxmax(),
            "most_common_freq": col_data.value_counts().max(),
            "unique_values": col_data.unique().tolist()
        }
    else:
        desc = {
            "dtype": str(dtype),
            "summary": "Unsupported or unknown type"
        }

    return {col: desc}

def decode_byte_columns(df):
    """
    Decode all byte string values in categorical (object) columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with potential byte-string values.

    Returns:
        pd.DataFrame: A new DataFrame with byte strings decoded to UTF-8 strings where applicable.
    """
    df_clean = df.copy()

    for col in df_clean.select_dtypes(include=['object']).columns:
        if isinstance(df_clean[col].iloc[0], bytes):
            df_clean[col] = df_clean[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df_clean

def class_to_int(class_col):
    """
    Convert a string-based binary class column to integer values 0 and 1.

    Args:
        class_col (Iterable): A column or list-like object with values '0' and '1' as strings.

    Returns:
        list: A list of integers where '0' becomes 0 and all other values become 1.
    """
    return [0 if x == '0' else 1 for x in class_col]

def clean_prep_modeling(df, seed, verbose = False):
    """
    Clean and prepare a dataset for modeling, including:
    - Column summaries
    - Byte decoding
    - Class conversion
    - One-hot encoding
    - Train/test splitting

    Args:
        df (pd.DataFrame): Input raw dataset
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing Xtrain, Xtest, yTrain, ytest
    """
    
    if verbose:
        ## initiate empty dictionary
        descriptions = {}

        ## get info about columns and print it to console
        for column in df.columns:
            descriptions.update(describe_column(df, column))
        
        ## suppress unless debug is on
        pprint.pprint(descriptions, verbose = False)

    ### remove the byte encoding
    df = decode_byte_columns(df)

    ### set the class as an int
    df['class'] = class_to_int(df['class'])

    ### one-hot encode the data
    categorical_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    ### train test split the data
    train, test = train_test_split(df, test_size = .80, random_state=seed)

    ## initialize output dict
    output = {}
    output['yTrain'] = train['class']
    output['ytest'] = test['class']
    output['Xtrain'] = train.drop('class', axis = 1)
    output['Xtest'] = test.drop('class', axis = 1)

    return output


if __name__ == '__main__':
    data, meta = arff.loadarff(SEISMIC_DATA)

    seis = pd.DataFrame(data = data, columns = meta.names())

    output = clean_prep_modeling(seis, SEED)

    write_pickle(CLEAN_DATA_DIR, 'modeling_data.pkl', output)
