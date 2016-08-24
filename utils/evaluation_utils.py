import numpy as np
import pandas as pd

def rmsle(y_pred,y_actual):
    return np.sqrt(np.mean(np.square(np.log(y_pred+1)-np.log(y_actual+1))))

def rmsle_on_logs(y_pred,y_actual):
    return np.sqrt(np.mean(np.square(y_pred-y_actual)))

def rmsle_score_on_logs(y_pred,y_actual):
    return np.maximum(1-np.sqrt(np.mean(np.square(y_pred-y_actual))),0)

def log_pandas(x):
    return np.log(x+1)

def inv_log_pandas(x):
    return np.exp(x)-1