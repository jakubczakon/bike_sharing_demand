from copy import copy

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class ExtractColumns(BaseEstimator, TransformerMixin):
    def __init__(self,colnames):
        self.colnames = colnames
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):

        X = X[self.colnames]
        return X   
    
class CleanCategoricalSimple(BaseEstimator, TransformerMixin):
    def __init__(self,colnames):
        self.colnames = colnames
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        for var in self.colnames:
            X_out[var] = X[var]-1
        return X_out 
    
    
class CleanCategorical(BaseEstimator, TransformerMixin):
    def __init__(self,colname):
        self.colname = colname
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        self.label_encoder.fit(X[self.colname])
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_out[[self.colname]] = self.label_encoder.transform(X[self.colname])
        return X_out
    
    
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self,colnames):
        self.colnames = colnames
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        
        X = X.drop(self.colnames, axis=1) 
        return X


class ExtractTimes(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        
        X["datetime"] =  pd.to_datetime(X["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        X["time_hour"] = X["datetime"].apply(lambda x: x.hour)
        X["date_weekday"] = X["datetime"].apply(lambda x: x.weekday())
        X["date_month"] = X["datetime"].apply(lambda x: x.month)
        X["date_day"] = X["datetime"].apply(lambda x: x.day)
        X["date_year"] = X["datetime"].apply(lambda x: x.year)
        X["day_night"] = X['time_hour'].apply(lambda x: is_day_or_night(x))
        
        return X
    
class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,colname):
        self.colname = colname
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        
    def fit(self, X, y=None):
        self.one_hot_encoder.fit(X[self.colname].values.reshape(-1, 1))
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_one_hot_encoded = self.one_hot_encoder.transform(X[self.colname].values.reshape(-1, 1))

        new_categorical_varnames = []
        for k in self.one_hot_encoder.active_features_:
            new_var = self.colname+"_%s"%k
            new_categorical_varnames.append(new_var)
        for i,var in enumerate(new_categorical_varnames):
            X_out[var] = X_one_hot_encoded[:,i]
        
        return X_out
   

class PandasCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,colname):
        self.colname = colname
        self.cnt_vct = CountVectorizer()
        
    def fit(self, X, y=None):
        self.cnt_vct.fit(X[self.colname].values.reshape(-1, 1))
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_one_hot_encoded = self.cnt_vct.transform(X[self.colname].values.reshape(-1, 1)).todense()

        print vocabulary_
        '''
        new_categorical_varnames = []
        for k in self.one_hot_encoder.active_features_:
            new_var = self.colname+"_%s"%k
            new_categorical_varnames.append(new_var)
        for i,var in enumerate(new_categorical_varnames):
            X_out[var] = X_one_hot_encoded[:,i]
        '''
        return X_out

class EncodeCategoricalDictVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,colnames):
        self.colnames = colnames
        self.category_encoder = DictVectorizer(sparse=False)
        
    def fit(self, X, y=None):
        categorical_variables = X[self.colnames].astype(str)
        df = categorical_variables.convert_objects(convert_numeric=True)
        self.category_encoder.fit(df.to_dict(orient='records'))
        return self
    
    def transform(self, X, y=None, copy=None):
        categorical_variables = X[self.colnames].astype(str)
        df = categorical_variables.convert_objects(convert_numeric=True)

        categorical_variables_transformed = self.category_encoder.transform(df.to_dict(orient='records'))
           
        for i,var in enumerate(self.colnames):
            X[var] = categorical_variables_transformed[:,i]
            
        return X
    
class PandasLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,colname):
        self.colname = colname
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        self.label_encoder.fit(X[self.colname].values)
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_out[self.colname] = self.label_encoder.transform(X[self.colname].values)            
        return X_out
    
class AddTimeGaps():
    def __init__(self,lags):
        self.lags = lags
    
    def transform(X):
        return X
    
def is_day_or_night(hour,day_limits = [6,21]):
    if hour>day_limits[0] and hour<day_limits[1]:
        return 1
    else:
        return 0
    
def preprocess_vanila(data,mode= "train"):
    data = data.sort_values(["datetime"])
    
    if mode =="train":
        X = data.drop(["count","registered","casual","datetime"], axis=1)
        Y = data[["count"]].values.ravel()
    
        return (X,Y)
    
    elif mode =="test":
        
        X = data.drop(["datetime"], axis=1)
        Index = data[["datetime"]]

        return (X,Index)
    
    else:
        
        raise Exception("Wrong mode")
        
def preprocess_weekdays(data,mode= "train"):
    data = data.sort_values(["datetime"])
    
    if mode =="train":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["count","registered","casual","datetime"], axis=1)
        Y = data[["count"]].values.ravel()
        
        return (X,Y)
    
    elif mode =="test":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["datetime"], axis=1)
        Index = data[["datetime"]]

        return (X,Index)
    
    else:
        
        raise Exception("Wrong mode")

def add_edge_gap(data,lag):
    result = data
    result["time_diff"] = result["datetime"].diff()
    result["time_diff"] = pd.to_numeric(result["time_diff"])/3600000000000
    result["max_in_lag"] = pd.Series.rolling(result["time_diff"],window=lag).max()
    result["max_in_lag"] = result["max_in_lag"].fillna(30)
    result["edge_lag_%s"%lag] = result["max_in_lag"].apply(lambda x:x>24)
    result["edge_lag_%s"%lag] =  result["edge_lag_%s"%lag]*1.0
    result.drop(["time_diff","max_in_lag"],axis=1)
    
    return result
        
def preprocess_rolling_edge(X_train,rolling_lags):
    for roll in rolling_lags: 
        X_train = add_edge_gap(X_train,lag=roll)
    return X_train

def preprocess_rolling(X_train,important_variables,rolling_lags=None,variables_to_roll=None):

    if not rolling_lags:
        rolling_lags = [5,12,24,50,100]
    elif not variables_to_roll:    
        variables_to_roll = ["temp","humidity","windspeed"]

    for roll in rolling_lags:

    #     X_train = lagged_day_night_mean(X_train,colnames = variables_to_roll,lag=roll,mode="day")
    #     X_train = lagged_day_night_mean(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        X_train = lagged_day_night_median(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        X_train = lagged_day_night_median(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        X_train = lagged_day_night_max(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        X_train = lagged_day_night_max(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        X_train = lagged_day_night_min(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        X_train = lagged_day_night_min(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        X_train = lagged_day_night_quantile_25(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        X_train = lagged_day_night_quantile_25(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        X_train = lagged_day_night_quantile_75(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        X_train = lagged_day_night_quantile_75(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        #X_train = lagged_day_night_quantile_variation(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        #X_train = lagged_day_night_quantile_variation(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        X_train = lagged_day_night_span(X_train,colnames = variables_to_roll,lag=roll,mode="day")
        X_train = lagged_day_night_span(X_train,colnames = variables_to_roll,lag=roll,mode="night")
        
    X_train = X_train[important_variables]
    
    return X_train

def preprocess_datetime(data,mode= "train"):
    data = data.sort_values(["datetime"])
    
    if mode =="train":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_day"] = data["datetime"].apply(lambda x: x.day)
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["count","registered","casual"], axis=1)
        Y = data[["count"]].values.ravel()
        
        return (X,Y)
    
    elif mode =="test":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_day"] = data["datetime"].apply(lambda x: x.day)
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data
        Index = data[["datetime"]]

        return (X,Index)
    
    else:
        
        raise Exception("Wrong mode")
                

def preprocess_weekdays_important(data,mode= "train"):
    data = data.sort_values(["datetime"])
    
    if mode =="train":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["count","registered","casual",
                       "datetime","weather","season",
                       "windspeed","holiday"], axis=1)
        Y = data[["count"]].values.ravel()
        
        return (X,Y)
    
    elif mode =="test":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["datetime","weather","season",
                       "windspeed","holiday"], axis=1)
        Index = data[["datetime"]]

        return (X,Index)
    
    else:
        
        raise Exception("Wrong mode")
        
def preprocess_weekdays_important_strict(data,mode= "train"):
    data = data.sort_values(["datetime"])
    
    if mode =="train":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["count","registered","casual",
                       "datetime","weather","season",
                       "windspeed","holiday","humidity",
                       "date_weekday","date_month"], axis=1)
        Y = data[["count"]].values.ravel()
        
        return (X,Y)
    
    elif mode =="test":
        
        data["datetime"] =  pd.to_datetime(data["datetime"], 
                                           format="%Y-%m-%d %H:%M:%S")
        data["time_hour"] = data["datetime"].apply(lambda x: x.hour)
        data["date_weekday"] = data["datetime"].apply(lambda x: x.weekday())
        data["date_month"] = data["datetime"].apply(lambda x: x.month)
        data["date_year"] = data["datetime"].apply(lambda x: x.year)
        
        X = data.drop(["datetime","weather","season",
                       "windspeed","holiday","humidity",
                       "date_weekday","date_month"], axis=1)
        Index = data[["datetime"]]

        return (X,Index)
    
    else:
        
        raise Exception("Wrong mode")
        
def generate_timelag_data(data,lag = 3):
    final = data
    lagging_set = data.drop(["time_hour","date_weekday",
                             "date_month","date_year",
                             "workingday","holiday","season"],axis = 1)
    for i in range(1,lag,1):
        x = lagging_set
        x = x.shift(i)
        x.rename(columns=lambda x: x+"_lag_%s"%i, inplace=True)
        final = pd.concat([final,x], axis=1)
    final = final.fillna(method="bfill")
   
    return final

def generate_mean_timelag_data(data,colnames,lag = 20):
    result = data
    for var_name in colnames:
        var_name_new = var_name+"_mean_whole_day_lag_%s"%lag
        result[var_name_new] = pd.Series.rolling(result[var_name],window = lag,min_periods=1).mean()
        result = result.fillna(method="bfill")
    return result

def generate_median_timelag_data(data,colnames,lag = 20):
    result = data
    for var_name in colnames:
        var_name_new = var_name+"_median_whole_day_lag_%s"%lag
        result[var_name_new] = pd.Series.rolling(result[var_name],window = lag,min_periods=1).mean()
        result = result.fillna(method="bfill")
    return result

def generate_min_timelag_data(data,colnames,lag = 20):
    result = data
    for var_name in colnames:
        var_name_new = var_name+"_min_whole_day_lag_%s"%lag
        result[var_name_new] = pd.Series.rolling(result[var_name],window = lag,min_periods=1).min()
        result = result.fillna(method="bfill")
    return result

def generate_max_timelag_data(data,colnames,lag = 20):
    result = data
    for var_name in colnames:
        var_name_new = var_name+"_max_whole_day_lag_%s"%lag
        result[var_name_new] = pd.Series.rolling(result[var_name],window = lag,min_periods=1).max()
        result = result.fillna(method="bfill")
    return result

def lagged_day_night_mean(data,colnames,lag,mode = "day"):
    trans_name = "mean"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=1).mean()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(full_result[colname])
    return full_result

def lagged_day_night_median(data,colnames,lag,mode = "day"):
    trans_name = "median"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:

        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=1).median()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(full_result[colname])
    return full_result

def lagged_day_night_max(data,colnames,lag,mode = "day"):
    trans_name = "max"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=1).max()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(full_result[colname])
    return full_result

def lagged_day_night_min(data,colnames,lag,mode = "day"):
    trans_name = "min"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=1).min()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(full_result[colname])
    return full_result

def lagged_day_night_quantile_25(data,colnames,lag,mode = "day"):
    trans_name = "quantile_25"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=4).quantile(0.25)
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(full_result[colname])
    return full_result

def lagged_day_night_quantile_75(data,colnames,lag,mode = "day"):
    trans_name = "quantile_75"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=4).quantile(0.75)
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(full_result[colname])
    return full_result

def lagged_day_night_span(data,colnames,lag,mode = "day"):
    trans_name = "span"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                var_max = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=lag).max()
                
                var_min = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=lag).min()
                
                gp[new_colname] = var_max-var_min
                
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(0)
    return full_result

def lagged_day_night_quantile_variation(data,colnames,lag,mode = "day"):
    trans_name = "quantile_variation"
    data["day_night"] = data['time_hour'].apply(lambda x: is_day_or_night(x))
    if mode=="day":
        mode_value=1
        
    elif mode=="night":
        mode_value=0
    else:
        raise Exception("Wrong mode")
        
    full_result = data.set_index("datetime",drop=False)
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        g= data.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                var_max = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=lag).quantile(0.75)
                
                var_min = pd.Series.rolling(gp[colname],
                                                       window=lag,
                                                       min_periods=lag).quantile(0.25)
                
                gp[new_colname] = var_max-var_min
                
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        full_result[new_colname] = colname_result[new_colname]

    full_result = full_result.sort_values(["datetime"]).fillna(method="ffill")
    for colname in colnames:
        new_colname = colname+"_%s_%s_%s"%(mode,trans_name,lag)
        full_result[new_colname] = full_result[new_colname].fillna(0)
    return full_result
