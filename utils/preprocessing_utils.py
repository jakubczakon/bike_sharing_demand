from copy import copy

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from statistics import mode


class ExtractColumns(BaseEstimator, TransformerMixin):
    def __init__(self,colnames):
        self.colnames = colnames
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):

        X = X[self.colnames]
        return X   
    
    
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
        X = X.sort_values(["datetime"])
        
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
    
    
class AddTimeGaps(BaseEstimator, TransformerMixin):
    
    def __init__(self,lag):
        self.lag = lag
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_out["time_diff"] = X["datetime"].diff()
        X_out["time_diff"] = pd.to_numeric(X_out["time_diff"])/3600000000000
        X_out["max_in_lag"] = pd.Series.rolling(X_out["time_diff"],window=self.lag).max()
        X_out["max_in_lag"] = X_out["max_in_lag"].fillna(30)
        X_out["edge_lag_%s"%self.lag] = X_out["max_in_lag"].apply(lambda x:x>24)
        X_out["edge_lag_%s"%self.lag] =  X_out["edge_lag_%s"%self.lag]*1.0
        X_out = X_out.drop(["time_diff","max_in_lag"],axis=1)           
        return X_out
 

class LaggingValues(BaseEstimator, TransformerMixin):
    
    def __init__(self,colname,lag):
        self.colname = colname
        self.lag = lag
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()

        for i in range(1,self.lag,1):
            x = X.copy()[[self.colname]]
            x = x.shift(i)
            x.rename(columns=lambda x: x+"_lag_%s"%i, inplace=True)
            X_out = pd.concat([X_out,x], axis=1)
        X_out = X_out.fillna(method="bfill")          
        return X_out
    

class LaggingMedian(BaseEstimator, TransformerMixin):
    
    def __init__(self,colname,lag,mode):
        self.colname = colname
        self.lag = lag
        self.mode = mode
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        trans_name = "median"
        X_out = X.copy()
        
        if self.mode =="whole":
            new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
            X_out[new_colname] = pd.Series.rolling(X_out[self.colname],window = self.lag,min_periods=1).median()
            X_out = X_out.sort_values(["datetime"])
            X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
            return X_out
        
        elif self.mode=="day":
            mode_value=1

        elif self.mode=="night":
            mode_value=0
        else:
            raise Exception("Wrong mode")

        X_out = X_out.set_index("datetime",drop=False)

        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        g= X_out.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[self.colname],
                                                           window=self.lag,
                                                           min_periods=1).median()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        X_out[new_colname] = colname_result[new_colname]

        X_out = X_out.sort_values(["datetime"])
        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
        return X_out

    
    
class LaggingMax(BaseEstimator, TransformerMixin):
    
    def __init__(self,colname,lag,mode):
        self.colname = colname
        self.lag = lag
        self.mode = mode
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        trans_name = "max"
        X_out = X.copy()
        
        if self.mode =="whole":
            new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
            X_out[new_colname] = pd.Series.rolling(X_out[self.colname],window = self.lag,min_periods=1).max()
            X_out = X_out.sort_values(["datetime"])
            X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
            return X_out
        
        elif self.mode=="day":
            mode_value=1

        elif self.mode=="night":
            mode_value=0
        else:
            raise Exception("Wrong mode")

        X_out = X_out.set_index("datetime",drop=False)

        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        g= X_out.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[self.colname],
                                                           window=self.lag,
                                                           min_periods=1).max()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        X_out[new_colname] = colname_result[new_colname]

        X_out = X_out.sort_values(["datetime"])
        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
        return X_out

    
class LaggingMin(BaseEstimator, TransformerMixin):
    
    def __init__(self,colname,lag,mode):
        self.colname = colname
        self.lag = lag
        self.mode = mode
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        trans_name = "min"
        X_out = X.copy()
        
        if self.mode =="whole":
            new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
            X_out[new_colname] = pd.Series.rolling(X_out[self.colname],window = self.lag,min_periods=1).min()
            X_out = X_out.sort_values(["datetime"])
            X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
            return X_out
        
        elif self.mode=="day":
            mode_value=1

        elif self.mode=="night":
            mode_value=0
        else:
            raise Exception("Wrong mode")

        X_out = X_out.set_index("datetime",drop=False)

        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        g= X_out.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[self.colname],
                                                           window=self.lag,
                                                           min_periods=1).min()
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        X_out[new_colname] = colname_result[new_colname]

        X_out = X_out.sort_values(["datetime"])
        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
        return X_out

def safe_mode(x):
    try:
        return mode(x)
    except Exception:
        return np.nan
    
    
class LaggingMode(BaseEstimator, TransformerMixin):
    
    def __init__(self,colname,lag,mode):
        self.colname = colname
        self.lag = lag
        self.mode = mode
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        trans_name = "mode"
        X_out = X.copy()

        if self.mode =="whole":
            new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
            X_out[new_colname] = pd.Series.rolling(X_out[self.colname],
                                                   window = self.lag,min_periods=1).apply(lambda x: safe_mode(x))
            X_out = X_out.sort_values(["datetime"])
            X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
            return X_out
        
        elif self.mode=="day":
            mode_value=1

        elif self.mode=="night":
            mode_value=0
        else:
            raise Exception("Wrong mode")

        X_out = X_out.set_index("datetime",drop=False)

        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        g= X_out.set_index('day_night', append=True,drop=False).groupby(level=1)
        colname_result = pd.DataFrame()
        for k, gp in g:
            if k==mode_value:
                gp[new_colname] = pd.Series.rolling(gp[self.colname],
                                                           window=self.lag,
                                                           min_periods=1).apply(lambda x: safe_mode(x))
            colname_result = pd.concat([colname_result,gp])
        colname_result = colname_result.set_index("datetime",drop=False)
        X_out[new_colname] = colname_result[new_colname]

        X_out = X_out.sort_values(["datetime"])
        new_colname = self.colname+"_%s_%s_%s"%(self.mode,trans_name,self.lag)
        X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
        return X_out

    
def is_day_or_night(hour,day_limits = [6,21]):
    if hour>day_limits[0] and hour<day_limits[1]:
        return 1
    else:
        return 0


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
