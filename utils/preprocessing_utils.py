from copy import copy

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor 
from sklearn.cross_validation import train_test_split

from statistics import mode

from evaluation_utils import rmsle, log_pandas, inv_log_pandas


class DataToTimeStamps(BaseEstimator, TransformerMixin):
    def __init__(self,stamp_size,dim_out):
        self.stamp_size = stamp_size
        self.dim_out = dim_out
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_out = pd.DataFrame(X_out)
        X = pd.DataFrame(X)
        stamp_list = []
        for i in range(self.stamp_size):
            x = X.copy()
            x = x.shift(i)
            x = x.fillna(0)
            stamp_list.append(x) 
        X_out = np.stack(stamp_list)
        X_out = X_out.transpose(1,0,2)
        if self.dim_out==4:
            n,t,v = X_out.shape
            X_out = X_out.reshape((n,t,1,v))
        
        return X_out 


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
    
class BinVariable(BaseEstimator, TransformerMixin):
    def __init__(self,bins,colname,drop_column = True):
        self.bins = bins
        self.colname = colname
        self.drop_column = drop_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_out[self.colname+"_binned"] = pd.cut(X[self.colname],bins = self.bins,include_lowest=True)
        if self.drop_column:
            X_out = X_out.drop([self.colname],axis=1)
        return X_out
    
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
    def __init__(self,colname,drop_colname=True):
        self.colname = colname
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.drop_colname = drop_colname
        
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
        
        if self.drop_colname:
            X_out = X_out.drop([self.colname],axis=1)
        
        return X_out
    
class DateToNumber(BaseEstimator, TransformerMixin):
    def __init__(self,colname):
        self.colname = colname
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, copy=None):
        X_out = X.copy()
        X_out[self.colname] = pd.to_numeric(X[self.colname])/3600000000000./24./30.
        return X_out.astype(int)
    
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
            X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
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
        X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
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
            X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
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
        X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
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
            X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
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
        X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
        X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
        return X_out
    
    
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
            X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
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
        X_out[new_colname] = X_out[new_colname].fillna(method = "bfill")
        X_out[new_colname] = X_out[new_colname].fillna(X_out[self.colname])
        return X_out


class RandomForestFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,n_estimators,drop_rate,feature_threshold,max_error_increase):
        self.n_estimators = n_estimators
        self.drop_rate = drop_rate
        self.feature_threshold = feature_threshold
        self.max_error_increase = max_error_increase
        
    def fit(self, X, y=None):
        self.important_features = X.columns.values
        self.last_rmsle = 10
        features = X.shape[1]
        while (features>self.feature_threshold):
            print "Number of features:",features
            X_tr,X_ts,Y_tr,Y_ts = train_test_split(X,y,train_size=0.8)
            
            rf = RandomForestRegressor(n_estimators=self.n_estimators,n_jobs=3)
            rf.fit(X_tr,Y_tr)
            
            Y_pred_valid_count = pd.DataFrame(rf.predict(X_ts)).apply(inv_log_pandas)
            Y_ts = pd.DataFrame(Y_ts)
            current_error = rmsle(Y_pred_valid_count.values.ravel(),
                                       Y_ts.apply(inv_log_pandas).values.ravel())
            
            error_change = current_error - self.last_rmsle
            print error_change
            if error_change > self.max_error_increase:
                print "last",self.last_rmsle
                print "current",current_error
                
                break
            else:
                self.last_rmsle = current_error
            
                importances = pd.DataFrame(dict(features = X.columns.values,
                                                importance =rf.feature_importances_)) 
                importances = importances.sort_values(["importance"],ascending=False)
                X = X[importances["features"].values[:-self.drop_rate]]
                features = X.shape[1]
                self.important_features = X.columns.values
                print "current error:",current_error
                
        return self
    
    def transform(self, X, y=None, copy=None):        
        return X[self.important_features]

def safe_mode(x):
    try:
        return mode(x)
    except Exception:
        return np.nan
    
def is_day_or_night(hour,day_limits = [6,21]):
    if hour>day_limits[0] and hour<day_limits[1]:
        return 1
    else:
        return 0