import numpy as np
import pandas as pd

def calculate_woe(df, feature, target):
    # Calculate the WoE for the feature using binning
    binned = pd.cut(df[feature], bins=10)  # 10 bins for demonstration
    df['binned_' + feature] = binned
    
    # Calculate good (1) and bad (0) distributions
    good = df[df[target] == 1].groupby('binned_' + feature, observed=False).size()
    bad = df[df[target] == 0].groupby('binned_' + feature, observed=False).size()
    
    # Calculate WoE for each bin
    good_dist = good / good.sum()
    bad_dist = bad / bad.sum()
    
    woe = np.log(good_dist / bad_dist)
    
    # Add WoE to dataframe
    df['woe_' + feature] = df['binned_' + feature].map(woe)
    
    return df, woe