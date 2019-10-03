import time
import pickle
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utilities import cal_score, convert_types

# Columns
cols_num = ['txn_dt', 'total_floor', 'building_complete_dt', 'parking_area', 'parking_price', 'txn_floor',
            'land_area', 'building_area', 'village_income_median', 'town_population', 'town_area',
            'town_population_density', 'doc_rate', 'master_rate', 'bachelor_rate', 'jobschool_rate', 'highschool_rate',
            'junior_rate', 'elementary_rate', 'born_rate', 'death_rate', 'marriage_rate', 'divorce_rate',
            'N_50', 'N_500', 'N_1000', 'N_5000', 'N_10000', 'lat', 'lon',
            'I_10', 'I_50', 'I_100', 'I_250', 'I_500', 'I_1000', 'I_5000', 'I_10000', 'I_MIN',
            'II_10', 'II_50', 'II_100', 'II_250', 'II_500', 'II_1000', 'II_5000', 'II_10000', 'II_MIN',
            'III_10', 'III_50', 'III_100', 'III_250', 'III_500', 'III_1000', 'III_5000', 'III_10000', 'III_MIN',
            'IV_10', 'IV_50', 'IV_100', 'IV_250', 'IV_500', 'IV_1000', 'IV_5000', 'IV_10000', 'IV_MIN',
            'V_10', 'V_50', 'V_100', 'V_250', 'V_500', 'V_1000', 'V_5000', 'V_10000', 'V_MIN',
            'VI_50', 'VI_100', 'VI_250', 'VI_500', 'VI_1000', 'VI_5000', 'VI_10000', 'VI_MIN',
            'VII_10', 'VII_50', 'VII_100', 'VII_250', 'VII_500', 'VII_1000', 'VII_5000', 'VII_10000', 'VII_MIN',
            'VIII_10', 'VIII_50', 'VIII_100', 'VIII_250', 'VIII_500', 'VIII_1000', 'VIII_5000', 'VIII_10000', 'VIII_MIN',
            'IX_10', 'IX_50', 'IX_100', 'IX_250', 'IX_500', 'IX_1000', 'IX_5000', 'IX_10000', 'IX_MIN',
            'X_10', 'X_50', 'X_100', 'X_250', 'X_500', 'X_1000', 'X_5000', 'X_10000', 'X_MIN',
            'XI_10', 'XI_50', 'XI_100', 'XI_250', 'XI_500', 'XI_1000', 'XI_5000', 'XI_10000', 'XI_MIN',
            'XII_10', 'XII_50', 'XII_100', 'XII_250', 'XII_500', 'XII_1000', 'XII_5000', 'XII_10000', 'XII_MIN',
            'XIII_10', 'XIII_50', 'XIII_100', 'XIII_250', 'XIII_500', 'XIII_1000', 'XIII_5000', 'XIII_10000', 'XIII_MIN',
            'XIV_10', 'XIV_50', 'XIV_100', 'XIV_250', 'XIV_500', 'XIV_1000', 'XIV_5000', 'XIV_10000', 'XIV_MIN']

cols_cat = ['building_material', 'city', 'building_type', 'building_use', 'parking_way', 'town', 'village']
# !!! Use target-mean encoding !!! #

cols_bin = ['I_index_50', 'I_index_500', 'I_index_1000', 'II_index_50', 'II_index_500', 'II_index_1000',
            'III_index_50', 'III_index_500', 'III_index_1000', 'IV_index_50', 'IV_index_500', 'IV_index_1000', 'IV_index_5000',
            'V_index_50', 'V_index_500', 'V_index_1000', 'VI_10', 'VI_index_50', 'VI_index_500', 'VI_index_1000',
            'VII_index_50', 'VII_index_500', 'VII_index_1000', 'VIII_index_50', 'VIII_index_500', 'VIII_index_1000',
            'IX_index_50', 'IX_index_500', 'IX_index_1000', 'IX_index_5000', 'X_index_50', 'X_index_500', 'X_index_1000',
            'XI_index_50', 'XI_index_500', 'XI_index_1000', 'XI_index_5000', 'XI_index_10000', 'XII_index_50', 'XII_index_500', 'XII_index_1000',
            'XIII_index_50', 'XIII_index_500', 'XIII_index_1000', 'XIII_index_5000', 'XIII_index_10000', 'XIV_index_50', 'XIV_index_500', 'XIV_index_1000']
# !!! Use target-mean encoding !!! #

cols_catbin = cols_cat + cols_bin

cols_feats = cols_num + cols_cat + cols_bin 

n_lencodes = 5

col_target = 'total_price'
col_target_log1p = 'target'


# Read data
df = pd.read_csv('dataset/train.csv', dtype=object)

# Preprocessing
### Convert types
df = convert_types(df, cols_num, col_target=col_target)

### Generate feats (train-test-same feats)
###create_feats(df)


# Feat Engineering
from feature_engineering import CategoricalColumnsEncoder
from feature_engineering import TargetMeanEncoding
from feature_engineering import MultiLabelEncoding

class FeatureEngineering():
    def __init__(self, cols_cat, cols_bin, n_lencodes):
        self.cols_catbin = cols_cat + cols_bin
        self.n_lencodes = n_lencodes
       
        self.cat_encoder = None
        self.target_encoder = None
        self.multi_lencoder = None

    def fit_transform(self, df):
        df = df.copy()

        self.target_encoder = TargetMeanEncoding(col_target = 'target')
        self.multi_lencoder = MultiLabelEncoding(self.n_lencodes)
        self.cat_encoder = CategoricalColumnsEncoder(mode='pandas')
    
        encoded1 = self.target_encoder.fit_transform(df[self.cols_catbin], df['target'])
        encoded2 = self.multi_lencoder.fit_transform(df[self.cols_catbin])
        self.cat_encoder.fit_transform(df, self.cols_catbin)

        return pd.concat([df, encoded1, encoded2], axis=1)

    def transform(self, df):
        df = df.copy()
        
        encoded1 = self.target_encoder.transform(df[self.cols_catbin])
        encoded2 = self.multi_lencoder.transform(df[self.cols_catbin])
        self.cat_encoder.transform(df)

        return pd.concat([df, encoded1, encoded2], axis=1)

# Grid search
# pars
is_log1p = True # if train on log1p target
# pars

# grid search
params_fix = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'mse',
              'metric': 'mape',
              'learning_rate': 0.100,
              }

# round 1
t0 = time.time()

params_gsearch1 = {'num_leaves': [63, 255, 511],           # may reduce in dim-reduction exp
                   'feature_fraction': [0.5, 0.75, 1.0],
                   'min_data_in_leaf': [5, 20, 50]
                   }

gsearch = {}
cols_feats_backup = cols_feats
folds = KFold(n_splits=3, shuffle=True, random_state=123)
for i_fold, (itrain, ival) in enumerate(folds.split(df)): # kfold
    print('==== Fold', i_fold+1, '====')

    # split train, val
    df_train = df.loc[itrain]
    df_val = df.loc[ival]

    # feat eng
    feat_eng = FeatureEngineering(cols_cat, cols_bin, n_lencodes)
    df_train = feat_eng.fit_transform(df_train)
    df_val = feat_eng.transform(df_val)
    cols_feats = cols_feats_backup + \
                 ['encoded_'+c for c in cols_catbin] + \
                 [f'lencoded_{i}_{c}' for i in range(n_lencodes) for c in cols_catbin]

    # Construct lgb dataset
    if is_log1p:
        lgb_train = lgb.Dataset(df_train[cols_feats], df_train['target']).construct()
        lgb_val = lgb.Dataset(df_val[cols_feats], df_val['target'], reference=lgb_train).construct()
    else:
        lgb_train = lgb.Dataset(df_train[cols_feats], df_train['total_price']).construct()
        lgb_val = lgb.Dataset(df_val[cols_feats], df_val['total_price'], reference=lgb_train).construct()

    # grid search
    for values in itertools.product(*[params_gsearch1[key] for key in params_gsearch1]):
        params = params_fix.copy()
        params.update( dict(zip(params_gsearch1.keys(), values)) )
        print('params:', params)

        model = lgb.train(params, lgb_train,
                          num_boost_round=10000,
                          valid_sets=lgb_val,
                          verbose_eval=2000,
                          early_stopping_rounds=200)
        y_pred = model.predict(df_val[cols_feats])

        if is_log1p:
            y_pred_expm1 = np.expm1(y_pred)
            y_pred_final = np.clip(y_pred_expm1, 0, None)
        else:
            y_pred_final = y_pred

        score = cal_score(df_val['total_price'], y_pred_final)
        tuple_params = tuple(params.items())
        gsearch[tuple_params] = gsearch.get(tuple_params, []) + [score]

# aggregate, sort gsearch results
gsearch_results1 = [[key, np.mean(val), val] for key, val in gsearch.items()]
gsearch_results1.sort(key= lambda x: x[1], reverse=True)
print(gsearch_results1)

print(f'total computing time: {time.time()-t0}')

# round 2
t0 = time.time()

params_gsearch2 = {'lambda_l1': [0, 0.01, 0.1],
                   'lambda_l2': [0, 0.01, 0.1]
                  }

gsearch = {}
folds = KFold(n_splits=3, shuffle=True, random_state=246)
for i_fold, (itrain, ival) in enumerate(folds.split(df)): # kfold
    print('==== Fold', i_fold+1, '====')

    # split train, val
    df_train = df.loc[itrain]
    df_val = df.loc[ival]

    # feat eng
    feat_eng = FeatureEngineering(cols_cat, cols_bin, n_lencodes)
    df_train = feat_eng.fit_transform(df_train)
    df_val = feat_eng.transform(df_val)
    cols_feats = cols_feats_backup + \
                 ['encoded_'+c for c in cols_catbin] + \
                 ['Lencoded_{i}_{c}' for i in range(n_lencodes) for c in cols_catbin]

    # Construct lgb dataset
    if is_log1p:
        lgb_train = lgb.Dataset(df_train[cols_feats], df_train['target']).construct()
        lgb_val = lgb.Dataset(df_val[cols_feats], df_val['target'], reference=lgb_train).construct()
    else:
        lgb_train = lgb.Dataset(df_train[cols_feats], df_train['total_price']).construct()
        lgb_val = lgb.Dataset(df_val[cols_feats], df_val['total_price'], reference=lgb_train).construct()

    # grid search
    # pick top 5 params from round 1
    for result1 in gsearch_results1[:3]:
        params1 = dict(result1[0])
        for values in itertools.product(*[params_gsearch2[key] for key in params_gsearch2]):
            params = params1.copy()
            params.update( dict(zip(params_gsearch2.keys(), values)) )
            print('params:', params)

            model = lgb.train(params, lgb_train,
                              num_boost_round=10000,
                              valid_sets=lgb_val,
                              verbose_eval=2000,
                              early_stopping_rounds=200)
            y_pred = model.predict(df_val[cols_feats])

            if is_log1p:
                y_pred_expm1 = np.expm1(y_pred)
                y_pred_final = np.clip(y_pred_expm1, 0, None)
            else:
                y_pred_final = y_pred

            score = cal_score(df_val['total_price'], y_pred_final)
            tuple_params = tuple(params.items())
            gsearch[tuple_params] = gsearch.get(tuple_params, []) + [score]

# aggregate, sort gsearch results
gsearch_results2 = [[key, np.mean(val), val] for key, val in gsearch.items()]
gsearch_results2.sort(key= lambda x: x[1], reverse=True)
print(gsearch_results2)

print(f'total computing time: {time.time()-t0}')

