import numpy as np
import pandas as pd

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

def generate_smart_timelag_data(data,lag = 3):
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