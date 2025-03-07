# src/feature_engineering.py
from imblearn.over_sampling import SMOTE
import pandas as pd
import datetime as dt

def create_time_features(df, time_col='TransactionStartTime'):
    """Creates time-based features from a datetime column."""
    df['TransactionHour'] = df[time_col].dt.hour
    df['TransactionDayOfWeek'] = df[time_col].dt.dayofweek
    df['TransactionMonth'] = df[time_col].dt.month
    df['TransactionYear'] = df[time_col].dt.year
    return df

def create_product_amount_interaction(df):
    """Creates an interaction feature between product category and mean amount."""
    product_category_means = df.groupby('ProductCategory')['Amount'].mean().to_dict()
    df['ProductCategory_Amount_Mean'] = df['ProductCategory'].map(product_category_means)
    return df

def create_customer_aggregate_features(df, customer_id_col='CustomerId', amount_col='Amount'):
    """Creates aggregate features per customer."""
    customer_agg = df.groupby(customer_id_col).agg({
        amount_col: ['sum', 'mean', 'count', 'std']
    })
    customer_agg.columns = ['customer_total_amount', 'customer_mean_amount', 'customer_transaction_count', 'customer_amount_std']
    df = pd.merge(df, customer_agg, on=customer_id_col, how='left')
    return df

import pandas as pd

def create_rfms_features(df, customer_id_col='CustomerId', transaction_time_col='TransactionStartTime', amount_col='Amount'):
    """Creates RFM features per customer."""
    df['TransactionDate'] = pd.to_datetime(df[transaction_time_col]).dt.date
    recent_date = pd.to_datetime(df['TransactionDate'].max())
    rfm_recency = df.groupby(customer_id_col)['TransactionDate'].max().reset_index()
    rfm_recency['TransactionDate'] = pd.to_datetime(rfm_recency['TransactionDate'])
    rfm_recency['Recency'] = (recent_date - rfm_recency['TransactionDate']).dt.days
    rfm_frequency = df.groupby(customer_id_col)[transaction_time_col].count().reset_index()
    rfm_frequency.columns = [customer_id_col, 'Frequency']
    rfm_monetary = df.groupby(customer_id_col)[amount_col].sum().reset_index()
    rfm_monetary.columns = [customer_id_col, 'Monetary']
    rfm = rfm_recency.merge(rfm_frequency, on=customer_id_col).merge(rfm_monetary, on=customer_id_col)
    df = pd.merge(df, rfm, on=customer_id_col, how='left')

    # Conditional drop
    if 'TransactionDate' in df.columns:
        df = df.drop(columns=['TransactionDate'])

    return df

def encode_categorical_features(df, categorical_cols, target_col='FraudResult'):
    """Encodes categorical features using target encoding."""
    for col in categorical_cols:
        if col in df.columns:
            means = df.groupby(col)[target_col].mean().to_dict()
            df[col + '_encoded'] = df[col].map(means)
            df = df.drop(col, axis=1)
    return df

def apply_smote(df, target_col='FraudResult'):
    """Applies SMOTE oversampling to balance the target variable."""
    smote = SMOTE(random_state=42)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_col)], axis=1)
    return df_resampled