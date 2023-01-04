import numpy as np

def psd2time(n, prms, fs):
    '''
    Generates time domain noise realization (y) from arbitary PSD.

    '''
    
    #n has to be even
    wn = prms.wn/1000 #[mv]
    c = prms.c/1000 # to match mv definitions
    alpha = prms.alpha
    
    dt = 1/fs
    time = np.arange(0,n)*dt    
    f2 = fs/2
    df = fs/n
    freq = np.linspace(df, f2, int(n/2))
    
    c = c*(alpha**2)
    psd = c/freq**alpha + wn*2 

    x1 = np.random.randn(len(freq)) + 1j*np.random.randn(len(freq))
    w1 = np.sqrt(psd/2)*x1
    x2 = np.random.randn(len(freq)) + 1j*np.random.randn(len(freq))
    w2 = np.sqrt(psd/2)*x2
    w = np.hstack( ( w1, np.conj(w2)[::-1] ))
    y = np.sqrt(fs)*np.sqrt(n)*np.real(np.fft.ifft(w))
    
    return freq, psd, time, y

