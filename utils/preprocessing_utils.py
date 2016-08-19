import numpy as np

def preprocess_vanila_random_forest(data,mode= "train"):
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
        
def get_hour(datetime):
    return
def get_day_of_the_week(datetime):
    return
def get_month(datetime):
    return
def get_year(datetime):   
    return
        
def preprocess_weekdays_random_forest(data,mode= "train"):
    if mode =="train":
        X = data.drop(["count","registered","casual"], axis=1)
        Y = data[["count"]].values.ravel()
        X["hour"] = X["datetime"]
        X["day"] = X["datetime"]
        X["month"] = X["datetime"]
        X["year"] = X["datetime"]
        
        return (X,Y)
    elif mode =="test":
        X = data.drop(["datetime"], axis=1)
        Index = data[["datetime"]]

        return (X,Index)
    else:
        raise Exception("Wrong mode")