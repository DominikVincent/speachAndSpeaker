# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
from scipy.signal import convolve, lfilter, hamming
from scipy.fftpack import fft
from lab1_tools import *
from scipy.fftpack.realtransforms import dct
from scipy.spatial.distance import cdist
def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    #print(samples.shape)
    #print(winlen)
    #print(winshift)
    length = int( np.ceil((samples.shape[0] - winlen)/winshift) )
    #print(length)
    count = 0
    arr = np.zeros((length, winlen))
    #print(arr.shape)
    for i in range(0, samples.shape[0] - winlen, winshift):
        arr[count] = samples[i:i+winlen]
        count += 1
    return arr
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    b = [1, -p]
    a = [1, 0]
    return lfilter(b, a, input)
    #return convolve(input, [1, -p])

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    return input*hamming(input.shape[1], sym=False)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.power(abs(fft(input, nfft)), 2)

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    filterbank = trfbank(samplingrate, 512)
    N,_ = input.shape
    M,_ = filterbank.shape
    result = np.zeros((N,M))
    for m in range(M):
        result[:,m] = np.dot(input, filterbank[m,:])
    return np.log(result)

    #filterbank = trfbank(samplingrate, 512)
    #arr = np.zero(input.shape[0], filterbank.shape[0])
    #for m in range(filterbank.shape[0]):



def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    mfcc = dct(input, n=nceps)
    #mfcc = mfcc[:,0:nceps]
    #return lifter(mfcc)
    return mfcc

def dtw(x, y, dist="euclidean"):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function passing a string instead: ----nope(can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    distances = cdist(x, y, dist)
    #print(distances)
    #print("test")
    #print(distances.shape)
    acc_distance = np.zeros_like(distances)

    h, w = distances.shape
    path = np.zeros((h,w,2), dtype=np.int32)

    # left, top, diag
    options = np.array([[1,0], [0,1], [1,1]])
    for y_ind in range(len(x)):
        for x_ind in range(len(y)):
            if x_ind>0 and y_ind>0:
                optionkkk = np.argmin( [acc_distance[y_ind - 1, x_ind], acc_distance[y_ind, x_ind - 1], acc_distance[y_ind - 1, x_ind- 1]])
                #print("option: ", optionkkk)
                path[y_ind, x_ind] = np.array([y_ind, x_ind]) - options[optionkkk]
                acc_distance[y_ind, x_ind] = distances[y_ind, x_ind] + np.min([acc_distance[y_ind -1, x_ind], acc_distance[y_ind, x_ind - 1], acc_distance[y_ind - 1, x_ind- 1]])
            elif x_ind == 0 and y_ind>0:
                path[y_ind, x_ind] = np.array([y_ind, x_ind]) - options[0]
                acc_distance[y_ind, x_ind] = distances[y_ind, x_ind] + acc_distance[y_ind -1, x_ind]
            elif x_ind > 0 and y_ind==0:
                path[y_ind, x_ind] = np.array([y_ind, x_ind]) - options[1]
                # print(distances[y_ind, x_ind])
                # print(acc_distance[y_ind, x_ind-1])
                acc_distance[y_ind, x_ind] = distances[y_ind, x_ind] + acc_distance[y_ind, x_ind-1]
            elif x_ind == 0 and y_ind==0:
                path[0,0] = [0,0]
                acc_distance[0,0] = distances[0,0]
            else:
                print("uncovered case with y", y_ind, " xind ", x_ind)
    fin_path = []
    tmp = [len(x)-1, len(y)-1]
    fin_path.append(tmp)
    while not (tmp[0] == 0 and tmp[1] == 0):
        #print(path[tmp[0], tmp[1]])
        fin_path.append(path[tmp[0], tmp[1]])
        tmp = path[tmp[0], tmp[1]]
    fin_path.reverse()
    return acc_distance[-1,-1]/(len(x)+len(y)), distances, acc_distance, np.array(fin_path)