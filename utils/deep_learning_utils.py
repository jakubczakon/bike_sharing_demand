import numpy as np

from theano import tensor as T
import theano

from sklearn.metrics import mean_squared_error

from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping

from evaluation_utils import rmsle

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
class ReportRmsleError(Callback):
    def __init__(self,X_callback,y_callback):
        self.X_callback = X_callback
        self.y_callback = y_callback
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = self.model.predict(self.X_callback)
        y_pred = np.exp(y_pred)-1
        y_true = np.exp(self.y_callback)-1

        print "RMSLE error:",rmsle(y_pred,y_true)