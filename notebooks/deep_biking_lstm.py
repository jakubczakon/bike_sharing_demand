
# coding: utf-8

# In[1]:

# from __future__ import absolute_import
import sys
sys.path.append("../")

import os
from copy import copy

import numpy as np
import pandas as pd

from matplotlib import pylab as plt
import seaborn as sns

from random import choice

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten,Convolution1D,Convolution2D,LSTM
from keras.regularizers import l2
from keras.optimizers import Adadelta,Adagrad,Adam,Adamax,RMSprop

from scipy.stats import randint as sp_randint

from utils.evaluation_utils import rmsle,log_pandas,inv_log_pandas
from utils.generic_utils import pickle_out,pickle_in
from utils.deep_learning_utils import ReportRmsleError
import utils.preprocessing_utils as prep



# In[4]:

(X_train,Y_train) = pickle_in(os.path.join("../","datasets","generated_features","train_binned.pkl"))
(X_valid,Y_valid) = pickle_in(os.path.join("../","datasets","generated_features","valid_binned.pkl"))
(X_test,Y_test) = pickle_in(os.path.join("../","datasets","generated_features","test_binned.pkl"))


# In[5]:

mms = MinMaxScaler(feature_range=(0, 1))
X_train = mms.fit_transform(X_train)
X_valid = mms.transform(X_valid)
X_test = mms.transform(X_test)

stp_gen = prep.DataToTimeStamps(stamp_size=10,dim_out=3)
X_train = stp_gen.fit_transform(X_train)
X_valid = stp_gen.fit_transform(X_valid)
X_test = stp_gen.fit_transform(X_test)

Y_train = Y_train.apply(log_pandas).values.ravel()
Y_valid = Y_valid.apply(log_pandas).values.ravel()
# Y_train = Y_train.values.ravel()
# Y_valid = Y_valid.values.ravel()

print X_train.shape,X_valid.shape


# In[6]:

l2_reg = 0.0000000000001
l2_reg_dense = 0.0000000000001

bike_lstm_model = Sequential()
bike_lstm_model.add(LSTM(10, input_dim=27))
bike_lstm_model.add(Dense(256))
bike_lstm_model.add(Dense(1))

bike_lstm_model.compile(loss ="mse",
                  optimizer = Adagrad())


# In[8]:

rms = ReportRmsleError(X_valid,Y_valid)
bike_lstm_model.fit(X_train, Y_train,
               validation_data=[X_valid,Y_valid], 
               batch_size=128, nb_epoch=10000, verbose=1, 
               callbacks=[rms],
               shuffle=False, class_weight=None, sample_weight=None)


# In[ ]:

