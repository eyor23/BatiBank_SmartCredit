import numpy as np
import pandas as pd

def calculate_woe_iv(dataset, feature, target):
    """Calculates Weight of Evidence (WoE) and Information Value (IV)."""
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset.groupby(feature, observed=True)[target].value_counts().reset_index(name='val')[dataset.groupby(feature, observed=True)[target].value_counts().reset_index(name='val')[feature] == dataset[feature].unique()[i]]['val'])
        try:
            lst.append([dataset[feature].unique()[i], val[0], val[1], val[0] / dataset[dataset[target] == 0].shape[0], val[1] / dataset[dataset[target] == 1].shape[0]])
        except IndexError:
            if len(val) == 1 and dataset[target].unique()[0] == 0:
                lst.append([dataset[feature].unique()[i], val[0], 0, val[0] / dataset[dataset[target] == 0].shape[0], 1e-10])
            elif len(val) == 1 and dataset[target].unique()[0] == 1:
                lst.append([dataset[feature].unique()[i], 0, val[0], 1e-10, val[0] / dataset[dataset[target] == 1].shape[0]])

    data = pd.DataFrame(lst, columns=[feature, 'Good', 'Bad', 'Good %', 'Bad %'])
    data['Good %'] = data['Good %'] + 0.0001
    data['Bad %'] = data['Bad %'] + 0.0001
    data['WoE'] = np.log(data['Good %'] / data['Bad %'])
    data['IV'] = (data['Good %'] - data['Bad %']) * data['WoE']
    data = data.sort_values(by=['IV'], ascending=False)
    data.replace([np.inf, -np.inf], 10, inplace=True)
    return data

def apply_woe_binning(dataset, feature_bins, target, woe_df):
    """Applies WoE binning to a dataset."""
    woe_map = dict(zip(woe_df[feature_bins], woe_df['WoE']))
    dataset[feature_bins + '_woe'] = dataset[feature_bins].map(woe_map)
    return dataset