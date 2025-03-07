import numpy as np
import pandas as pd

def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[target],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[target],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[target]
        })

    dset = pd.DataFrame(lst)
    dset['Good %'] = (dset['Good'] + 0.000001) / dset['All']
    dset['Bad %'] = (dset['Bad'] + 0.000001) / dset['All']
    dset['WoE'] = np.log(dset['Good %'] / dset['Bad %'])
    dset = dset.replace([np.inf, -np.inf], 1000000) #replace inf with large number.
    dset['IV'] = (dset['Good %'] - dset['Bad %']) * dset['WoE']
    iv = dset['IV'].sum() #calculate IV sum outside of the loop.
    return dset

def apply_woe_binning(dataset, feature, target, woe_df):
    woe_dict = woe_df.set_index('Value')['WoE'].to_dict()
    dataset[feature + '_woe'] = dataset[feature].map(woe_dict)
    return dataset