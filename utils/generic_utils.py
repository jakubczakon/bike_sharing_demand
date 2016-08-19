import cPickle as pickle

def pickle_in(datapath):
    with open(datapath,"rb") as f:
        data = pickle.load(f)
    return data

def pickle_out(datapath,data):
    with open(datapath,"wb") as f:
        pickle.dump(data,f)