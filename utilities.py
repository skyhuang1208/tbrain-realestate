import numpy as np
import pandas as pd
from IPython.display import display

def cal_mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / y_true)

def cal_score(y_true, y_pred):
    rate = np.mean(np.abs(y_pred - y_true) <= 0.1 * y_true)
    mape = np.mean(np.abs(y_pred - y_true) / y_true)
    return np.round(rate, decimals=4)*10000 + (1-mape)

def sigmoid_01(x):
    return 1 / (1 + np.exp(-50*(x-0.1))) + 0.1*x

def cal_score_smooth(y_true, y_pred):
    pe = np.abs(y_pred - y_true) / y_true
    return 100*np.mean(sigmoid_01(pe))

def convert_types(df, cols_num, cols_date=[], col_target=None):
    # num feats
    df[cols_num] = df[cols_num].astype('float32')

    # dt feats
    for c in cols_date:
        df[c] = pd.to_datetime(df[c])

    if col_target is not None:
        df[col_target] = df[col_target].astype('float32')
        df['target'] = np.log1p(df[col_target])
        df['price_per_area'] = df['total_price'] / df['building_area']
        df['log_price_per_area'] = np.log1p(df['price_per_area'])
        
    return df

def create_dt_feats(df, col_feat):
    df['dow_'+col_feat] = (df[col_feat] % 7).astype(int)
    df['day_in_year_'+col_feat] = (df[col_feat] % 365).astype(int)
    df['month_'+col_feat] = (df['day_in_year_'+col_feat] / 30.4167).map(np.floor).astype(int)
    df['quarter_'+col_feat] = (df['day_in_year_'+col_feat] / 91.25).map(np.floor).astype(int)
    df['year_'+col_feat] = (df[col_feat] / 365.25).map(np.floor).astype(int)

    return df

def check(array, n=5):
    split = len(array) > 2*n
    if type(array) == np.ndarray:
        if split:
            print(array[:n], array[-n:])
        else:
            print(array)
        print('shape =', array.shape, 'dtype =', array.dtype)
    elif type(array) == pd.core.frame.DataFrame or type(array) == pd.core.series.Series:
        if split:
            display(pd.concat([array.head(n), array.tail(n)]))
        else:
            display(array)
        print('shape =', array.shape)
    else:
        if split:
            print(array[:n], array[-n:])
        else:
            print(array)
        print('shape =', len(array), 'type =', type(array))
pd.core.frame.DataFrame.show = check
pd.core.frame.DataFrame.check = check
pd.core.series.Series.show = check
pd.core.series.Series.check = check
