# src/utils.py
import pandas as pd

def drop_missing_columns(df, columns):
    """Drops columns with all missing values."""
    df = df.drop(columns=columns, errors='ignore')
    return df

def drop_correlated_columns(df, columns):
    """Drops highly correlated columns."""
    df = df.drop(columns=columns, errors='ignore')
    return df