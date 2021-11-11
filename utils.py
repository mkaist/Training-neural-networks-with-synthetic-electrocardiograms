import yaml
import json
import numpy as np
import pickle
from scipy import signal
from wfdb.processing import normalize_bound
from artefact.data import load_noise, get_noise
from syntmodel.utils_model import Prms
from syntmodel.synt_gen import generate_randomset

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)   
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)       
    
    if args['verbose']:
        print('args: {}'.format(args))
    
    return dict2obj(args)

def load_pickle(fullpath):
  with open(fullpath, 'rb') as handle:
      return pickle.load(handle)  

def shift_labels(data, fixlabelint):
    '''Shifts given label to max in given window of length fixlabelint.
    Data is [(ecg, r_label)] : [(narr, list)]
    '''
    r_inds_new, ecg_new = [], []
    for item in data:
        ecg, r_inds = item        
        r_new = []
        for r in r_inds:
            start, stop = max(0, r - fixlabelint), min(len(ecg), r + fixlabelint)
            r_candidate = np.argmax(ecg[start:stop]) + start
            
            if ecg[r_candidate] > ecg[r]:
                r_new.append(r_candidate)                
            else:
                r_new.append(r) #dont do anything
        
        r_inds_new.append(r_new)
        ecg_new.append(ecg)

    data = list(zip(ecg_new, r_inds_new))
    return data


def training_gen(args):

    label_repeats = 5 #number of ones per r-wave fixed
    n_timesteps = int(args.preprocess.resample_fs*args.data.sample_time)  
    prms = Prms() 
    prms.scale_limits(args.parameters.scale_waveform, 
                  args.parameters.scale_fiducialpoint, 
                  args.parameters.scale_rr,
                  args.parameters.scale_noise)     

    def repeat(n, indices, repeats):  
         label = np.zeros(n)
         for indice in indices:
             label[indice - repeats : indice + repeats + 1] = 1 
         return label    

    if args.preprocess.data_augmentation:
        ma, bw = load_noise(args)
    
    while True:

        X, Y = [], []

        while len(X) < args.train.batch_size:            
                
            data, noise_streams = generate_randomset(args, prms)               
            data = shift_labels(data, 8)  
            data = add_noise(data, noise_streams)
            ecg, r_inds = data[0]
            label = repeat(len(ecg), r_inds, label_repeats)
            x = ecg
            y = label

            if args.preprocess.data_augmentation:
                x = normalize_bound(x, lb=-1, ub=1)
                x = x + get_noise(ma, bw, n_timesteps, int(args.preprocess.resample_fs))            

            if args.preprocess.filter:
                x = bpf(x, args.preprocess.resample_fs)   
   
            x = normalize_bound(x, lb=-1, ub=1)    

            X.append(x)
            Y.append(y)

        X = np.asarray(X)
        Y = np.asarray(Y)

        X = X.reshape(X.shape[0], X.shape[1], 1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).astype(int)

        yield (X, Y)   
        
def add_noise(data, noise_streams):
    ecg_new, r_inds_new = [], []     
    for (ecg, r_inds), noise in zip(data, noise_streams): 
        ecg = ecg + noise
        ecg_new.append(ecg)
        r_inds_new.append(r_inds)    
    data = list(zip(ecg_new, r_inds_new))
    return data

def bpf(arr, fs, lf=0.5, hf=50, order=2):
    wbut = [2*lf/fs, 2*hf/fs]
    sos = signal.butter(order, wbut, btype = 'bandpass', output = 'sos')       
    return signal.sosfiltfilt(sos, arr, padlen=250, padtype='even') 

def reshape(X_train, Y_train, n_samples, n_timesteps):
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape([n_samples,n_timesteps,1])
    X_train = np.array(X_train)
    X_train = X_train.reshape([n_samples,n_timesteps,1])     
    return X_train, Y_train 