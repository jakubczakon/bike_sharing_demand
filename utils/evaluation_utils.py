import numpy as np

def rmsle(y_pred,y_actual):
    return np.sqrt(np.mean(np.square(np.log(y_pred+1)-np.log(y_actual+1))))
