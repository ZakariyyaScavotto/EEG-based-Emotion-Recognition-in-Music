from pylsl import StreamInlet, resolve_stream #Used to read in EEG data from EmotivPRO software
from joblib import load #used to load the SVMs
import scipy.signal as signal #used for bandpass filtering, Welch's method
from scipy.stats import differential_entropy #used to calculate DE for feature reduction
import numpy as np

#REMEMBER: BEFORE RUNNING MAKE SURE THAT EMOTIV LSL IS RUNNING

def testMain():
    return 'Program executed'