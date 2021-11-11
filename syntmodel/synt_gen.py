import numpy as np
from syntmodel.signal_model import gen_rr, gen_phase, gen_ecg
from syntmodel.noise_model import psd2time


def model(n, prms, fs):
    ''' 
    Returns one ecg witn n r-peaks and corresponding noise stream 
    and r_indices. 
    '''
    
    rr = gen_rr(n, prms)    
    rr_phase = gen_phase(rr, fs)
    time, ecg_clean, label = gen_ecg(rr_phase, fs, prms=prms)
    _, _, _, noise_stream = psd2time(len(ecg_clean), prms, fs)
    return ecg_clean, noise_stream, label


def generate_randomset(args, prms):
    '''
    Generates a random set of ecgs where args (dict) holds generation input
    arguments and prms the signal properties..

    '''
                                          
    sample_len = args.syntdata.fs*args.data.sample_time    
   
    dataset, noiseset = [], []
    for i in range(args.data.n_samples):           
        prms.randomize() 
        #n_rri = 10, generate enough extra so it can be cropped to desired length  
        ecg_clean, noise_stream, (p,q,r,s,t) = model(10, prms, args.syntdata.fs) 
            
        all_ = (ecg_clean, (p,q,r,s,t), noise_stream)        
        data, noise_stream = crop_signal(all_, sample_len, args.syntdata.fs, prms.mu)     
        dataset.append(data)  
        noiseset.append(noise_stream)
         
    return dataset, noiseset


def crop_signal(signal, wlen, fs, random_start):
    '''
    Crops a signal and r-indices into spesific length (wlen) and 

    '''
    #use mu fron synt model for starting randomization       
    ecg, labels, noise_stream = signal     
    start = np.random.randint(0, int(fs*random_start)) 
    stop = start + wlen
    
    assert(len(ecg) > stop)
    ecg = ecg[start:stop]
    noise_stream = noise_stream[start:stop]
    
    _,_,r,_,_ = labels  
    r = r[(r >= start) & (r <= stop-1)] - start 

    return (ecg, r), noise_stream 
