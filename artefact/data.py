from wfdb import rdsamp
import numpy as np
from scipy.signal import resample_poly

#Source: https://github.com/jtlait/ecg2rr

def create_sine(fs, time_s, sine_frequency):
    """
    Create sine wave.
    Function creates sine wave of wanted frequency and duration on a
    given sampling frequency.
    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency used to sample the sine wave
    time_s : float
        Lenght of sine wave in seconds
    sine_frequency : float
        Frequency of sine wave
    Returns
    -------
    sine : array
        Sine wave
    """
    samples = np.arange(time_s * fs) / fs
    sine = np.sin(2 * np.pi * sine_frequency * samples)

    return sine

                                ##
def get_noise(ma, bw, win_size, fs):
    """
    Create noise that is typical in ambulatory ECG recordings.
    Creates win_size of noise by using muscle artifact, baseline
    wander, and mains interefence (60 Hz sine wave) noise. Windows from
    both ma and bw are randomly selected to
    maximize different noise combinations. Selected noise windows from
    all of the sources are multiplied by different random numbers to
    give variation to noise strengths. Mains interefence is always added
    to signal, while addition of other two noise sources varies.
    Parameters
    ----------
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    win_size : int
        Wanted noise length
    Returns
    -------
    noise : array
        Noise signal of given window size
    """
    # Get the slice of data
    beg = np.random.randint(ma.shape[0]-win_size)
    end = beg + win_size
    beg2 = np.random.randint(ma.shape[0]-win_size)
    end2 = beg2 + win_size

    # Added term 'fs'
    # Get mains_frequency US 60 Hz (alter strenght by multiplying)
    mains = create_sine(fs, int(win_size/fs), 60)*np.random.uniform(0, 0.5)

    # Choose what noise to add
    mode = np.random.randint(2)

    # Add noise with different strengths
    ma_multip = np.random.uniform(0, 5)
    bw_multip = np.random.uniform(0, 10)

    # Add noise
    if mode == 0:
        noise = ma[beg:end]*ma_multip
    elif mode == 1:
        noise = bw[beg:end]*bw_multip
    else:
        noise = (ma[beg:end]*ma_multip)+(bw[beg2:end2]*bw_multip)

    return noise+mains

def load_noise(args):
    # Load data
    baseline_wander = rdsamp('bw', pn_dir='nstdb')
    muscle_artifact = rdsamp('ma', pn_dir='nstdb')

    # Concatenate two channels to make one longer recording
    ma = np.concatenate((muscle_artifact[0][:,0], muscle_artifact[0][:,1]))
    bw = np.concatenate((baseline_wander[0][:,0], baseline_wander[0][:,1]))

    # Resample noise to wanted Hz
    ma = resample_poly(ma, up=int(args.preprocess.resample_fs), down=muscle_artifact[1]['fs'])
    bw = resample_poly(bw, up=int(args.preprocess.resample_fs), down=baseline_wander[1]['fs'])

    return ma,bw