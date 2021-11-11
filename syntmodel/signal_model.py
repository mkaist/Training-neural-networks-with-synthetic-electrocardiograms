
import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks


def rand_sequence(n, a, std):
    '''
    s: random stream drawn from power distribution
    x: random stream drawn from normal distribution
    '''
      
    s = np.random.pareto(a, n)*6 + 1
    x = np.random.randn(n)*std  

    return np.array(s, dtype=int), np.array(x)

def gen_rr(n, prms):
    '''
    n: number of samples (r-peak locations)    
    '''    
    
    breathing_gen = lambda bf, bc, br_prev: bc*np.sin(2*np.pi*br_prev*bf)
    y = stochastic(n, prms.a, prms.std, prms.b)    
   
    rr = np.zeros(n)
    
    for i in np.arange(n):  
        br_prev = np.sum(rr)
        rr[i] = prms.mu + breathing_gen(prms.bf, prms.bc, br_prev) + y[i]     
    return rr


def stochastic(n, a, std, b):

    #Source: Citation J. W. Kantelhardt et al 2003 EPL 62 147

    k, x = rand_sequence(n,a,std)    
    y, y_tmp = np.zeros(n), np.zeros(n)
    for i in np.arange(n):       
        #eq. 3          
        if i - k[i] > 0:
            avg = np.mean(np.square(y_tmp[i-k[i]:i])) 
            y_tmp[i] = x[i]*np.sqrt(1+b*avg)            
        else: 
            y_tmp[i] = 0   
            
        m = np.zeros(i)
        m[(k[:i] + np.arange(i) - i) > 0] = 1
        y[i] = 0.05*np.sum(y_tmp[:i]*m)
    return y

def gen_phase(rrs, fs):
    phase = []    
    for rr in rrs:          
        phase.append(np.linspace(-np.pi, np.pi, int(rr*fs)))
    return np.concatenate(phase).ravel()

def gen_ecg(phase, fs, prms):     
    '''Transforms phase signal to synthetic ecg with labels'''
    
    phase = phase.copy()
    def asym(arr,a,b,m):        
        c = lambda p, a, b, m: -(m*2*np.pi/np.square(b))*a*p*np.exp( -m*np.square(p)/(2*np.square(b)) )        
        neg_ind = np.where(arr<=0)
        pos_ind = np.where(arr>0)      
        arr[neg_ind] = c(arr[neg_ind],a,b,1)
        arr[pos_ind] = c(arr[pos_ind],a,b,m)
        return arr       
    
    p = np.roll(phase, int(fs*prms.pd))
    q = np.roll(phase, int(fs*prms.qd))
    r = np.roll(phase, int(fs*prms.rd))
    s = np.roll(phase, int(fs*prms.sd))
    t = np.roll(phase, int(fs*prms.td*np.sqrt(prms.mu))) #CHECK THIS
   
    dz =  asym(p, prms.pa, prms.pb, prms.pm) +\
          asym(q, prms.qa, prms.qb, prms.qm) +\
          asym(r, prms.ra, prms.rb, prms.rm) +\
          asym(s, prms.sa, prms.sb, prms.sm) +\
          asym(t, prms.ta, prms.tb, prms.tm)     
          
    #get event labels
    locs_max, _ = find_peaks(phase) 
    locs_min = locs_max + 1 
    locs_min = np.insert(locs_min, 0, 0)
    locs_max = np.insert(locs_max, len(locs_max), len(phase))    

    plabel, qlabel, rlabel, slabel, tlabel = [],[],[],[],[]
    for loc_min, loc_max in zip(locs_min, locs_max):        
        mid =  int(round((loc_min + loc_max)/2))  
        plabel.append(int(round(mid + prms.pd*fs)))
        qlabel.append(int(round(mid + prms.qd*fs)))
        rlabel.append(int(round(mid + prms.rd*fs)))
        slabel.append(int(round(mid + prms.sd*fs)))
        tlabel.append(int(round(mid + prms.td*fs)))
        
    label = (np.array(plabel), np.array(qlabel), np.array(rlabel),\
             np.array(slabel), np.array(tlabel))
    
    #ensure even number of samples
    z = cumtrapz(dz, dx=1/fs, initial=0)
    if np.mod(len(z),2) == 1:
        z = z[:-1]
    
    return np.arange(len(z))/fs, z, label


 
    