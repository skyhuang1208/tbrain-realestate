import math
import numpy as np
import pandas as pd

class CategoricalColumnsEncoder():
    '''Convert categorical feature columns in a pandas dataframe to number
       and store the mapping. Reduce the memory usage by lightgbm when construct lightgbm 
       dataset. Or, allowing feeded to embeddings of keras.
    '''
    def __init__(self, mode='pandas', reserve_missing=False):
        assert mode in ['lightgbm', 'keras', 'pandas']
        self.mode = mode
        self.reserve_missing = reserve_missing
        
    def fit_transform(self, df, cols_features_cat):
        assert all([c in df for c in cols_features_cat]), 'Not all columns found in the dataframe'
        self.cols_features_cat = cols_features_cat
        self.cat_dtypes = {}  # store the cat to code info for later prediction
        self.cat_min_value = {}
        self.cat_n_classes = {}   # store numbers of classes including null value for each cat for keras mode
        for c in cols_features_cat:
            df[c] = df[c].astype('category')
            self.cat_dtypes[c] = df[c].dtype
            
            if self.mode=='lightgbm':
                df[c] = df[c].cat.codes
                df[c] = df[c].astype('float32').replace(-1.0, np.nan)
            elif self.mode=='keras':
                df[c] = df[c].cat.codes
                if self.reserve_missing:
                    c_min = -1  # always keep the spot for missing value
                else:
                    c_min = df[c].min()
                self.cat_min_value[c] = c_min
                df[c] = df[c] - c_min  # so that cat number start from 0
                self.cat_n_classes[c] = df[c].max() + 1  # max class index + 1 for the output encode range
                
    def transform(self, df):
        assert all([c in df for c in self.cols_features_cat]), 'Not all columns found in the dataframe'
        for c in self.cols_features_cat:
            dtype = self.cat_dtypes[c]
            df[c] = df[c].astype(dtype)
            
            if self.mode=='lightgbm':
                df[c] = df[c].cat.codes
                df[c] = df[c].astype('float32').replace(-1.0, np.nan)
            elif self.mode=='keras':
                df[c] = df[c].cat.codes
                c_min = self.cat_min_value[c]
                df[c] = df[c] - c_min # so that cat number start from 0


class TargetMeanEncoding:
    def __init__(self, col_target='total_costs'):
        self.col_target = col_target
        self.dummy = 'M!ss!nG' # this not used; just avoid collision with others

        # target encoder, decoder
        self.target_encoder = {}
        self.target_decoder = {}

    def fit_transform(self, df, target):
        assert target.notnull().all(), '(cat fit_transform) target has nan'

        # make target encoder & decoder & avg_level
        avg_target_all = target.mean()

        for col in df: # use baysian mean with c=avg_count
            # get avg N samples for a val
            df_temp = pd.concat([df[col].fillna(self.dummy), target], axis=1) # new df
            avg_count = df_temp.shape[0] / df_temp[col].nunique()
            
            # get baysian mean for each val
            baysian_mean = lambda x: (np.sum(x) + avg_count*avg_target_all) \
                                     / (x.count() + avg_count)
            avg_target = df_temp.groupby(col)[self.col_target].agg(baysian_mean).reset_index()

            # add 'M!ss!nG' and 'MeAnlvl' into encoder
            # if NaN in data, 'M!ss!nG' in encoder, so we add 'MeAnlvl'
            # if not, add 'M!ss!nG' to encoder and later dup to 'MeAnlvl'
            bmean_all = baysian_mean(target) # baysian mean for whole samples
            if self.dummy in avg_target[col]:   # meanlvl != missinglvl; add meanlvl
                avg_target = avg_target.append(pd.DataFrame( # avg_all if no nan
                    [['MeAnlvl', bmean_all]], columns=[col, self.col_target]) )
            else:                               # meanlvl == missinglvl
                avg_target = avg_target.append(pd.DataFrame( # avg_all if no nan
                    [[self.dummy, bmean_all]], columns=[col, self.col_target]) )

            # sorting
            val_sorted = avg_target.sort_values(by=self.col_target).reset_index(drop=True)

            # get ints
            self.target_encoder[col] = dict(zip(val_sorted[col], val_sorted.index+1)) # +1: start at 1
            self.target_decoder[col] = dict(zip(val_sorted.index+1, val_sorted[col]))
            
            # dup 'MeAnlvl' to 'M!ss!nG'
            if self.dummy not in avg_target[col]:
                self.target_encoder[col]['MeAnlvl'] = \
                    self.target_encoder[col][self.dummy]

        return self.transform(df)
        
    def transform(self, df):
        assert (set(self.target_encoder.keys()) - set(df.columns))==set(), \
            '(cat transform) not found all columns in df'
                
        # target encoding
        df_tarenc = pd.DataFrame(index= df.index)
        for col in df:
            col_name = 'encoded_'+col

            arr = df[col].fillna(self.dummy) # new arr
            df_tarenc[col_name] = arr.map(self.target_encoder[col]) # unknown vals: meanlvl
            ratio_unknown = sum(df_tarenc[col_name].isnull())/df_tarenc.shape[0]
            if ratio_unknown > 0.7: # check
                print('Warning (tar_encode): unknown ratio for %s: %.5f' % (col_name, ratio_unknown))
            df_tarenc.loc[df_tarenc[col_name].isnull(), col_name] = self.target_encoder[col]['MeAnlvl']
        df_tarenc = df_tarenc.astype('int')
        
        return df_tarenc 
    
    def inverse_transform(self,df,col):
        raise Exception('inverse_transform of category not implemented')


class MultiLabelEncoding:
    def __init__(self, n_encodes=4):
        self.n_encodes = n_encodes
        self.n_cols = {}
        self.encoder = {}
        self.decoder = {}

    def fit_transform(self, df):
        for col in df:
            self.encoder[col] = []
            self.decoder[col] = []
            vals = list(df[col].unique()) + ['M!ss!nG', 'UnKnoWn']
            self.n_cols[col] = min(math.factorial( min(len(vals), 10) )/2, self.n_encodes)
            for i in range(self.n_cols[col]):
                rand_list = np.arange(len(vals))
                np.random.shuffle(rand_list)
                self.encoder[col].append(dict(zip(vals, rand_list)))
                self.decoder[col].append(dict(zip(rand_list, vals)))

        return self.transform(df)
        
    def transform(self, df):
        encoded = pd.DataFrame()
        for col in df:
            arr = df[col].fillna('M!ss!nG')
            for i in range(self.n_cols[col]):
                col_name = ('lencoded_%d_' % i) + col
                encoded[col_name] = arr.map(self.encoder[col][i])
                encoded[col_name] = encoded[col_name].fillna(self.encoder[col][i]['UnKnoWn'])

        return encoded
    
    def inverse_transform(self,df,col):
        raise Exception('inverse_transform of category not implemented')

class Imputation():
    ''' Fill missing values 
    fill rule:
        if in customized_map - use it
        else -  if skewness in [-1,1], fillna by mean
                else, fillna by median
    '''
    def __init__(self):
        self.impute_map = {}

    def fit_transform(self, df, cols=None, customized_map=None):
        from scipy import stats

        if cols is None:
            cols = df.columns

        map_tmp = {}
        for c in cols:
            if (customized_map is not None) and (c in customized_map):
                map_tmp[c] = customized_map[c]
            else:
                skewness = stats.skew(df[c])
                if -1 <= skewness <= 1: map_tmp[c] = 'mean'
                else:                   map_tmp[c] = 'median'
        
        for col, imp_method in map_tmp.items():
            if imp_method=='mean':      self.impute_map[col]= np.nanmean(df[col])
            elif imp_method=='median':  self.impute_map[col]= np.nanmedian(df[col])
            elif imp_method=='mode':    self.impute_map[col]= df[col].mode()[0]
            else:                       self.impute_map[col]= imp_method
        
        return self.transform(df,cols)

    def transform(self, df, cols=None):
        if cols is None:
            cols = df.columns
        df = df[cols].copy()

        for c in cols:
            df[c] = df[c].fillna(self.impute_map[c])

        return df[cols]

# Scaler for pandas dataframe ==============
class PdStandardScaler():
    '''Standardize features by removing the mean and scaling to unit variance
       Apply to selected columns of a pandas dataframe
       Examples:
           scaler = PdStandardScaler()
           df = pd.DataFrame([[1, 2, 3], [3, 4, 5]], columns=['a', 'b', 'c'])
           scaler.fit(df, cols=['a', 'b'])
           scaler.transform(df, cols_input=['a', 'b'], cols_transformed=['at', 'bt'])
           scaler.inverse_transform(df, cols_input=['at', 'bt'], 
                                    cols_transformed=['ao', 'bo'])
       Inspired by https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    '''
    def __init__(self): # , with_mean=True, with_std=True
        #self.with_mean = with_mean
        #self.with_std = with_std
        pass

    def fit(self, df, cols=None):
        '''Compute the mean and std of each cols in df to be used for later scaling.'''
        self.scale_ = []
        self.mean_ = []
        if cols is None:
            cols = list(df.columns)
        self.cols_ = cols
        for c in cols:
            self.mean_.append(df[c].mean())
            self.scale_.append(df[c].std())
    
    def transform(self, df, cols_input=None, cols_transformed=None):
        '''Transform'''
        if cols_input is None: 
            cols_input = self.cols_
        if cols_transformed is None:
            cols_transformed = cols_input
        for c_in, c_out, x_mean, x_scale in zip(cols_input, cols_transformed, self.mean_, self.scale_):
            df[c_out] = (df[c_in] - x_mean) / x_scale
    
    def inverse_transform(self, df, cols_input=None, cols_transformed=None):
        '''Inverse transform'''
        if cols_input is None: 
            cols_input = self.cols_
        if cols_transformed is None:
            cols_transformed = cols_input
        for c_in, c_out, x_mean, x_scale in zip(cols_input, cols_transformed, self.mean_, self.scale_):
            df[c_out] = x_scale * df[c_in] + x_mean

class PdMedianScaler():
    '''Standardize features by removing the median and scaling to unit variance
       Apply to selected columns of a pandas dataframe
       Examples:
           scaler = PdMedianScaler()
           df = pd.DataFrame([[1, 2, 3], [3, 4, 5]], columns=['a', 'b', 'c'])
           scaler.fit(df, cols=['a', 'b'])
           scaler.transform(df, cols_input=['a', 'b'], cols_transformed=['at', 'bt'])
           scaler.inverse_transform(df, cols_input=['at', 'bt'], 
                                    cols_transformed=['ao', 'bo'])
       Inspired by https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    '''
    def __init__(self): # , with_mean=True, with_std=True
        #self.with_mean = with_mean
        #self.with_std = with_std
        pass

    def fit(self, df, cols=None):
        '''Compute the mean and std of each cols in df to be used for later scaling.'''
        self.scale_ = []
        self.mean_ = []
        if cols is None:
            cols = list(df.columns)
        self.cols_ = cols
        for c in cols:
            self.mean_.append(df[c].median())  # use median not mean
            self.scale_.append(df[c].std())
    
    def transform(self, df, cols_input=None, cols_transformed=None):
        '''Transform'''
        if cols_input is None: 
            cols_input = self.cols_
        if cols_transformed is None:
            cols_transformed = cols_input
        for c_in, c_out, x_mean, x_scale in zip(cols_input, cols_transformed, self.mean_, self.scale_):
            df[c_out] = (df[c_in] - x_mean) / x_scale
    
    def inverse_transform(self, df, cols_input=None, cols_transformed=None):
        '''Inverse transform'''
        if cols_input is None: 
            cols_input = self.cols_
        if cols_transformed is None:
            cols_transformed = cols_input
        for c_in, c_out, x_mean, x_scale in zip(cols_input, cols_transformed, self.mean_, self.scale_):
            df[c_out] = x_scale * df[c_in] + x_mean

class PdMinMaxScaler():
    '''Transforms features by scaling each feature to a given range.
       Apply to selected columns of a pandas dataframe
       Inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    '''
    def __init__(self, transformed_range=(0, 1)):
        assert transformed_range[0] < transformed_range[1]
        self.t_min = transformed_range[0]
        self.t_max = transformed_range[1]
        self.t_range = self.t_max - self.t_min

    def fit(self, df, cols=None):
        self.scale_ = []
        self.min_ = []
        if cols is None:
            cols = list(df.columns)
        self.cols_ = cols
        for c in cols:
            c_min = df[c].min()
            c_max = df[c].max()
            c_range = c_max - c_min
            if c_range == 0.: 
                x_scale = self.t_range
            else:
                x_scale = self.t_range / c_range
            x_min = self.t_min - c_min * x_scale
            self.min_.append(x_min)
            self.scale_.append(x_scale)
    
    def transform(self, df, cols_input=None, cols_transformed=None):
        '''Transform'''
        if cols_input is None: 
            cols_input = self.cols_
        if cols_transformed is None:
            cols_transformed = cols_input
        for c_in, c_out, x_min, x_scale in zip(cols_input, cols_transformed, self.min_, self.scale_):
            df[c_out] = df[c_in]*x_scale + x_min
    
    def inverse_transform(self, df, cols_input=None, cols_transformed=None):
        '''Inverse transform'''
        if cols_input is None: 
            cols_input = self.cols_
        if cols_transformed is None:
            cols_transformed = cols_input
        for c_in, c_out, x_min, x_scale in zip(cols_input, cols_transformed, self.min_, self.scale_):
            df[c_out] = (df[c_in] - x_min) / x_scale
