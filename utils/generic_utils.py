import cPickle as pickle
from sklearn.externals import joblib

def pickle_in(datapath,compresion_mode=0):
    if compresion_mode ==0:
        with open(datapath,"rb") as f:
            data = pickle.load(f)
        return data
    else:
        data = joblib.load(datapath)
        return data
        
def pickle_out(datapath,data,compresion_mode = 0):
    if compresion_mode ==0:
        with open(datapath,"wb") as f:
            pickle.dump(data,f)
    else:
        joblib.dump(data, datapath, compress=compresion_mode)