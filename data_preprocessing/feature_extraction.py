#!/usr/bin/env python
# coding: utf-8

# # Feature Extraction Notebook - ChirPy
# This notebook details the methods used to pre-process audio data for the ChirPy classifier. All default values are specific to BirdCLEF 2021 kaggle challenge, but can be changed in the Global Config section
# 
# Written by Eric Gonzalez

# ### Import Packages

import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from scipy import stats
from copy import copy
import numpy as np
import logging
import pandas as pd
from scipy import signal
import scipy
from scipy import ndimage
from dateutil.parser import parse
from time import time
from datetime import date
from glob import glob
# logging options
#logging.basicConfig(filename='analysis_tools.log',filemode='w',level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


# 
# ### Global Configuration Variables

# In[2]:


# path to kaggle data folder, based on user system
rel_path = '../Bird Data/'

# path to location where feature extraction dataframes will be saved
save_path = rel_path+'processed_data/'
# create save path if it does not exist
if not os.path.exists(save_path): os.mkdir(save_path)

# paths relative to kaggle
metadata_path      = rel_path+'train_metadata.csv'
short_audio_path   = rel_path+'train_short_audio/'
long_audio_path    = rel_path+'train_soundscapes/'
long_metadata_path = rel_path+'train_soundscape_labels.csv'
long_loc_path      = rel_path+'test_soundscapes/'

# default values for various quantities
default_sample_rate  = 32000
default_n_fft_fine   = 2048
default_n_fft_coarse = 1024
default_audio_extension = ['.ogg']
default_alpha = 0.25
default_peak_pad = 0.15


# ## Data Loading Functions

# First we will define functions to read the audio data from a specified path. 

# In[3]:


# function for loading all audio clips in a directory
# default extension: .ogg
# audio_fpath is the relative or absolute path to the folder containing audio files of interest
def load_audio_files(audio_fpath,ext=default_audio_extension):
    #initialize list for holding audio files
    audio_files = []
    #recursively load audio, then filter based on specified extension(s)
    for root,dirs,files in os.walk(audio_fpath):
        for file in files:
            if file.endswith(tuple(ext)):
                audio_files.append(root+'/'+file)
    # retuns a list with path to all audio files with specified extension(s)
    return audio_files

# define a function to quickly and easily load audio files for testing/writing functions
def test_audio(short_path=short_audio_path,long_path=long_audio_path):
    #use function on both paths
    short_audio_files = load_audio_files(short_path)
    long_audio_files  = load_audio_files(long_path)
    return short_audio_files[:10], long_audio_files[:10]


# Now we can make use of the `librosa` package to easily load audio data from file path. 
# * We take the convention that `x` is the time series data and `X` is the frequency domain data. In general, time series data and its variations will be lower case while frequency domain data will be upper case.
# * The `smoothing` boolean enables smoothing via a gaussian filter. The `gsigma` option is the standard deviation of the gaussian kernel.
# * The parameter `sr` is the sample rate of the audio file. The default is $32,000\ \text{Hz}$.
# * When possible, time series data is windowed prior to a fourier transformation to suppress spectral leakage. A tukey window is chosen as it does not modify transient signals as much as other window choices [1,2].
# * The parameter `alpha` is a shape parameter, from the `scipy` documentation: "Shape parameter of the Tukey window, representing the faction of the window inside the cosine tapered region. If zero, the Tukey window is equivalent to a rectangular window. If one, the Tukey window is equivalent to a Hann window."
# * Returns the original, unwindowed time series data, `x`, the stft of the time data, `X`, and an array of the frequencies corresponding to `X`, `freqs`. We also return `window_function` in case we need to invert the transformation and obtain the data from inverting the fourier transform on `X`.

# In[4]:


# function for loading both time series data x and 
def librosa_load_both(audio_file,sr=default_sample_rate,              window='tukey',alpha=default_alpha,n_fft=default_n_fft_fine,smoothing=True,gsigma=1):
    #librosa takes audio clip path and loads it as a waveform array
    x, sr = librosa.load(audio_file,sr=sr)
    
    # perform gaussian smoothing
    x = ndimage.gaussian_filter1d(x,gsigma)
    
    # use window function to suppress spectral leakage
    if window == 'tukey':
        window_function = signal.windows.tukey(x.shape[0],alpha)
    elif window =='none':
    # option to use no is kept, in case it is needed
        window_function = 1
    else: 
        valid_windows = ['tukey','none']
        statement = """Window function \'{}\' has not been implemented or does not exist.
            Please select from {}""".format(window,valid_windows)
        raise ValueError(statement)

    # window the time series data
    x_win = x.copy()
    x_win *= window_function
    
    #librosa's short time ft
    X = librosa.stft(x_win,n_fft=n_fft)
    freqs = np.arange(0, 1 + n_fft / 2) * sr / n_fft
    return x, X, freqs, window_function


# One low level, but extremely useful feature is the power spectral density. The power spectral density $S(f)$ or `psd` is given by the diagonal terms of the covariance matrix, $C_{i,j}$ of the frequency domain data `X`. The following function returns the power spectrum of `X` along with the mean `mu` and variance `sigma` of said power spectrum.

# In[5]:


def compute_psd_data(X):
    Xcov = np.cov(X)
    psd = np.abs(np.diag(Xcov))
    mu = np.mean(psd)
    sigma = np.std(psd)
    return psd, mu, sigma


# ## Data Visualization I

# Basic audio visualization plots: Time Series Plot,  Spectrogram, Power Spectral Density, and Covariance Matrix

# In[41]:


def basic_plots(x,X,freqs,sr=default_sample_rate):
    # set up plotting canvas
    fig = plt.figure(figsize=(12,16))
    gs  = fig.add_gridspec(3,2,wspace=0.2,hspace=0.4)
    ax1 = fig.add_subplot(gs[0,:])
    # librosa has useful plotting functions
    # plot the time series data
    ax1.set_title('Waveplot of Time Series Data')
    librosa.display.waveplot(x,sr)
    
    # create a spectrogram
    ax2 = fig.add_subplot(gs[1,0])
    ax2.set_title('Spectrogram of Frequency Domain')
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='log')
    plt.colorbar(label='Amplitude (db)')
    
    # alongside the spectrogram, display the covariance matrix
    ax3 = fig.add_subplot(gs[1,1])
    ax3.set_title('Covariance Matrix of Frequency Domain Data')
    Xcov = np.cov(X)
    Xdbcov = librosa.amplitude_to_db(abs(Xcov))
    librosa.display.specshow(Xdbcov,sr=sr,x_axis='log',y_axis='log')
    plt.colorbar(label='Amplitude')
    
    #compute psd for plotting
    psd,psd_mu,psd_sigma = compute_psd_data(X)
    # plot the power spectrum
    ax4 = fig.add_subplot(gs[2,:])
    ax4.set_title('Power Spectral Density')
    ax4.plot(freqs,psd+1,c='k')
    ax4.set_xlabel('Frequency ($Hz$)')
    plt.show()


# ## Noise Reduction and Event Localization

# In principle, we can reduce the noise in our samples by estimating the power spectrum of the noise and using our estimation to remove the noise from the audio data by rescaling the frequency in each bin in `X` according to the amplitude of the noise in that bin. The noise reduced signal $X_\text{NR}$ with equally weighted bins is then [2]
# 
# $$X_\text{NR}(f)= \frac{X(f)}{\left[S(f)\right]^{1/2}}$$
# 
# The $0^\text{th}$ order estimation of the noise would be to take the power spectrum of the entire clip; for a long enough clip the noise would be dominant. This may work for the long $10\ \text{min}$ clips but the short training clips are on the order of $30-60\ \text{sec}$, in which case it is harder to justify.
# 
# The $1^\text{st}$ order correction to our estimate would be to remove any signal events from the clip, then take the power spectrum of the residual time series data. We write the following functions to make these estimates.

# ### Noise Reduction 

# First we can test the $0^\text{th}$ order estimate by taking clips and rescaling by the overall power. The following function performs this task and returns a new set nose reduced of time and frequency domain data, `x_NR` and `X_NR` respectively.

# In[64]:


def noise_reduction(x,X,window_function,sr=default_sample_rate,n_fft=default_n_fft_fine,filter_red=False,filter_red_val=128):
    # we use the fft from the windowed function to find the power spectrum
    psd,psd_mu,psd_sigma = compute_psd_data(X)
    # compute amplitude of noise in each bin
    noise_amp = np.sqrt(np.abs(psd))
    
    # use new instance of stft as X came from a windowed x
    X_NR = librosa.stft(x*window_function)
    # iterate over frequency bins
    for i,amplitude in enumerate(noise_amp):
        # divide all samples of a given frequency in the spectrogram by the psd amplitude
        # avoid amplifying bins with no power using min function
        X_NR[i,:] /= amplitude+1
    
    # invert fft onrescaled spectral data
    ## normalize time series data to unit variance
    #x_NR /= np.std(x_NR)
    # compute frequency based on n_ffts
    freqs = np.arange(0, 1 + n_fft / 2) * sr / n_fft
    
    if filter_red: 
        def filter_scaling(freq):
            return psd_mu*(np.exp(-((freq-filter_red_val)**2)/(2*filter_red_val**2)))
        
        for i,freq in enumerate(freqs):
            if freq < filter_red_val:
                X_NR[i,:] = filter_scaling(freq)
            elif freq >= filter_red_val:
               break
    
    x_NR = librosa.istft(X_NR)
    
    return x_NR,X_NR,freqs


# We can also define plotting functions to compare before and after the noise reduction.

# In[9]:


def plot_denoise_comparisons(x,X,x_NR,X_NR,sr=default_sample_rate):
        # set up plotting canvas
    fig = plt.figure(figsize=(12,16))
    gs  = fig.add_gridspec(4,2,wspace=0.2,hspace=0.4)
    
    # create waveplots
    ax1 = fig.add_subplot(gs[0,:])
    ax1.set_title('Waveplot of Time Series Data')
    librosa.display.waveplot(x,sr)
        
    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_title('Waveplot of Time Series Data after Denoising')
    librosa.display.waveplot(x_NR,sr)
    
    # create spectrograms
    ax3 = fig.add_subplot(gs[2,0])
    ax3.set_title('Spectrogram of Frequency Domain')
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='log')
    plt.colorbar(label='Amplitude (db)')
    
    ax4 = fig.add_subplot(gs[2,1])
    ax4.set_title('Spectrogram of Frequency Domain after Denoising')
    Xdb_NR = librosa.amplitude_to_db(abs(X_NR))
    librosa.display.specshow(Xdb_NR,sr=sr,x_axis='time',y_axis='log')
    plt.colorbar(label='Amplitude (db)')

    #compute psd for plotting
    psd,psd_mu,psd_sigma = compute_psd_data(X)
    psd_NR,psd_mu_NR,psd_sigma_NR = compute_psd_data(X_NR)
    # plot the power spectrum
    ax5 = fig.add_subplot(gs[3,:])
    ax5.set_title('Power Spectral Density with and without Denoising')
    ax5.plot(freqs,psd,c='k',label='with noise')
    ax5.plot(freqs,psd_NR,'k-.',label='denoised')
    ax5.set_xlabel('Frequency ($Hz$)')
    ax5.legend()
    plt.show()


# This method of noise reduction is simple and straightforward. Clearly the information in the power spectrum is erased in the denoised clip since all bins are weighted equally; the interesting information lies in the localized events. The denoising made peaks in the time series data more prominent, and this can help us perform the first order correction to the denoising. 

# ### Event Localization

# In order to identify events in the time series data, we can make use of the `scipy.signal` package, namely two methods `find_peaks` and `find_peaks_cwt`.
# 
# The `find_peaks` method finds all local maxima by comaring neighboring values. Peaks can be filtered by selecting optional arguments in the `find_peaks` method, of interest to us are
# * The `prominence` parameter, according to `scipy.signal` documentation is "how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line."
# * The `width` parameter can filter by requiring peaks to have a minimum width (and maximum width when two elements are specified), which is useful especially in the case where the width of the pulse can be approximated.
# 
# The `find_peaks_cwt` method uses a continuous wavelet transformation to determine the peak locations. This method convolves the time series data with a wavelet selected by the `wavelet` parameter with the wavelet size specified by `widths`. This method searches for ridge lines in originating from local maxima of wavelet coefficients at `widths` length scales, and accepts a peak if ridge lines indicate the relative maxima appears in successive length scales. We can perform a continuous wavelet transformation on the time series data to see the ridge lines and visualize the process.

# In[12]:


def cwt_visualization(x,wavelet=signal.ricker,widths=np.logspace(9,14,base=2,num=100)):
    cwt_matrix = signal.cwt(np.abs(x),wavelet=wavelet,widths=widths)
    plt.figure(figsize=(8,8))
    plt.imshow(cwt_matrix,aspect='auto',vmax=abs(cwt_matrix).max(),vmin=0)
    plt.show()


# In[13]:


#cwt_visualization(x_NR)


# The strategy for peak finding is to cross reference these two methods for agreement. First, we write a function to perform `find_peaks` on the time series data. We also write a function for visualization. In the peak finding function, we note:
# * Since the denoised data has prominent peak, the peaks will mostly be filtered by their `prominence`. The `prominence` threshold is `6*np.std(x)` to ensure we only catch signal peaks, if the noise is gaussian distributed then this should effectively filter out noise peaks.
# * The `width` parameter needs to be specified in order for the method to return the peak width. It is important to return the peak width as it will be used to estimate the event duration and perhaps determine the scale for `widths` in `find_peaks_cwt`. We don't really care about filtering by width, we expect prominence to filter, so we cast a wide net with `widths=[0.5,5000]`.

# In[14]:


def find_x_peaks(x,wavelet=signal.ricker,widths=[1,5000],prominence=None):
    # compute prominence, cannot set as default value due to x dependence
    if type(prominence) == type(None): prominence = np.std(x)*5
    # use method to find peaks, returns location of peak by index of x and the features of each peak.
    peaks,peak_features = signal.find_peaks(np.abs(x),width=widths,prominence=prominence)
    peak_features['indices'] = peaks
    return peaks,pd.DataFrame(peak_features)

def plot_x_peaks(x,sr=default_sample_rate,wavelet=signal.ricker,widths=[1,5000],prominence=None):
    # find peaks
    peaks,peak_features = find_x_peaks(x,wavelet=wavelet,widths=widths,prominence=prominence)
    # create time array for plotting
    time = np.arange(len(x))/sr
    
    plt.figure(figsize=(12,8))
    plt.plot(time,x)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Peaks Found using `find_peaks` Method')
    
    # plotting peak center and width on a line above the time series data
    peak_line_height = 1.1*x.max()
    
    for i,peak_f in peak_features.iterrows():
        # find index of lower bound of width
        # max between lb and 0 to ensure we don't return negative index
        # subtract 1 for narrow peaks so lower edge doesn't coincide with center
        lb = max(int(peak_f['indices'] - (peak_f['widths'])/2 - 1),0)
        # repeat with upper bound, similar logic
        ub = min(int(peak_f['indices'] + peak_f['widths']/2 + 2),len(x))
        # compute width
        wd = ub-lb
        # create line from width
        wd_line = np.ones(wd)*peak_line_height
        # now plot line in correct time window
        plt.scatter(time[lb:ub],wd_line,c='k')

    plt.show()


# In[15]:

# Now we can try the `find_peak_cwt` method. The method in question only returns peak locations, not peak features so we don't need to write a function. Instead, we directly compute and plot the peaks from `find_peak_cwt`.

# In[16]:


def plot_x_peaks_cwt(x,sr=default_sample_rate,widths=np.logspace(11,13,base=2,num=100)):
    
    # use method to find peaks, returns location of peak by index of x and the features of each peak.
    peaks = signal.find_peaks_cwt(x,widths=widths)
    
    # create time array for plotting
    time = np.arange(len(x))/sr
    
    plt.figure(figsize=(8,8))
    plt.plot(time,x)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # plotting peak center and width on a line above the time series data
    peak_points = np.ones(len(peaks))*1.1*x.max()
    plt.scatter(time[peaks],peak_points,c='k')
    
    plt.show()


# In[17]:


#plot_x_peaks_cwt(x_NR)


# It looks like the `find_peak_cwt` method is not suitable for this application, likely because the time series data has a relatively high sample rate. This causes the computation to take very long, so we will not be using it for this application.

# ## Noise and Signal Separation

# Now that we have developed a method for selecting signal peaks, we can use this to separate the noise from the signals. The following function separates the time series by selecting all data that is at $x_\text{peak}\pm x_\text{pad}$ with the default value $x_\text{pad}=0.2\ \text{ms}$ is set by `default_peak_pad = 0.2`. Since our peak features extraction functions also estimate the width, the duration corresponding to one Full Width Half Maximum (FWHM) is included in $x_\text{peak}$. The argument `peak_features` is the dataframe returned from the `find_x_features` function, allowing for flexibility in peak localization implementation (rather than just calling `find_x_features` inside the function).
# First we write a function to simply split the audio that is considered noise from the signal, without any additional processing.

# In[18]:



def split_signal_noise_audio(x,peak_features,sr=default_sample_rate,peak_pad=default_peak_pad):
    #convert peak pad units ms to time index 
    signal_half_duration = peak_pad*sr
    # peak time span list initialization
    peak_time_list = []
    for i,peak_f in peak_features.iterrows():
        # find index of lower bound of width
        # max between lb and 0 to ensure we don't return negative index
        lb = int(peak_f['indices'] - (peak_f['widths'])/2)
        lb -= signal_half_duration
        lb = max(lb,0)
        # repeat with upper bound, similar logic
        ub = int(peak_f['indices'] + peak_f['widths']/2)
        ub += signal_half_duration 
        ub = min(ub,len(x)-1)
        
        # with foresight for dropping repeated indexes, we will convert a numpy array of the range to a pandas series
        peak_time = np.arange(lb,ub)
        peak_time_list.append(peak_time.copy())
    peak_time_series = pd.Series(peak_time_list,name='peak times').explode().drop_duplicates()
    peak_time_series = peak_time_series.astype(int).values
    
    time = np.arange(0,len(x))*default_sample_rate
    signal = x[peak_time_series]
    
    noise_time_series = np.delete(np.arange(len(x)),peak_time_series)
    noise = x[noise_time_series]
    
    fig = plt.figure(figsize=(12,8))
    gs  = fig.add_gridspec(2,1,wspace=0.2,hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Waveplot of Time Series Signal Data')
    librosa.display.waveplot(signal,sr)
        
    ax2 = fig.add_subplot(gs[1,0])
    ax2.set_title('Waveplot of Time Series Noise Data')
    librosa.display.waveplot(noise,sr)
    return signal,noise
        



# There are still some calls in the background of the noise, we could in principle repeat the procedure but removing surious signals becomes much more difficult. We could also attempt to implement the `cwt` but in the interest of time we will skip it for now. Ultimately, this is a good approximation because we care about the overall features of the power spectrum with noise removed, with clean power spectral features the random forest will perform well.

# ### Noise Reduction Revisited

# In order to make use of the clipped data, the noise clips must be appropriately windowed otherwise we risk adding spectral leakage due to discontinuities in the clip. We will reuse some of the above code to find the time span of the noise for appropriate windowing. The following function does index gymnastics to massage the estimated peak locations into two `pandas` dataframes, `peaks_df` and `noise_df`. Both dataframes have pairs of indexes which correspond to the edges of the peaks and noise in the time series data. This allows us to clip the audio for analysis.

# In[153]:


def split_signal_noise(x,peak_features,sr=default_sample_rate,peak_pad=default_peak_pad):
    #convert peak pad units ms to time index 
    signal_half_duration = peak_pad*sr
    #initialize a dataframe to hold the peak bounds
    columns = ['lower bound','upper bound']
    peaks_df = pd.DataFrame(data=None,columns=columns)
    for i,peak_f in peak_features.iterrows():
        bound_dict = {}
        
        # find index of lower bound of width
        # max between lb and 0 to ensure we don't return negative index
        lb = int(peak_f['indices'] - (peak_f['widths'])/2)
        lb -= signal_half_duration
        lb = max(lb,0)
        bound_dict['lower bound'] = int(lb)
        # repeat with upper bound, similar logic
        ub = int(peak_f['indices'] + peak_f['widths']/2)
        ub += signal_half_duration 
        ub = min(ub,len(x)-1)
        bound_dict['upper bound'] = int(ub)
        peaks_df = peaks_df.append(bound_dict,ignore_index=True)
        
    # now that the df holds all the peak bounds, and is already organized in ascenting order,
    # we can merge peak windows
    # make array to hold indices to be dropped so we don't mess up the iteration
    to_drop_list = []
    
    for i,bounds in peaks_df[:-1].iterrows():
        #check for upper edge
        if bounds['upper bound'] < len(x)-1:
            if peaks_df.iloc[i+1]['lower bound'] <= bounds['upper bound'] and                              bounds['upper bound'] <= peaks_df.iloc[i+1]['upper bound']:
                #if conditions are met, the peaks overlap
                bounds['upper bound'] = peaks_df.iloc[i+1]['upper bound']
                to_drop_list.append(i)
    peaks_df = peaks_df.drop(to_drop_list,axis=0)
    peaks_df = peaks_df.reset_index().drop('index',axis=1)
    
    if not peaks_df.empty:
        noise_bottom_edge = pd.Series([0],name='lower bound')
        noise_upper_edge  = pd.Series([len(x)],name='upper bound')
        noise_lb = pd.Series(peaks_df['upper bound'],name='lower bound')
        noise_ub = pd.Series(peaks_df['lower bound'],name='upper bound')

        if noise_ub.iloc[0]==0:
            noise_ub = noise_ub[1:]
        else:
            noise_lb = noise_bottom_edge.append(noise_lb)

        if noise_lb.iloc[-1]==len(x):
            noise_lb = noise_lb[:-1]
        else:
            noise_ub = noise_ub.append(noise_upper_edge)
        
        noise_lb = noise_lb.reset_index().drop('index',axis=1)
        noise_ub = noise_ub.reset_index().drop('index',axis=1)
        noise_df = pd.concat([noise_lb,noise_ub],axis=1)
    
    elif peaks_df.empty:
        noise_dict = {}
        noise_dict['upper bound'] = len(x)
        noise_dict['lower bound'] = 0
        noise_df = pd.DataFrame(data=noise_dict)
    
    return peaks_df,noise_df
        
def characterize_noise_psd(x,peak_features,alpha=0.5):
    peak_df,noise_df = split_signal_noise(x,peak_features)
    
    x_clipped = x.copy()
    for i,bounds in peak_df.iterrows():
        x_clipped[bounds['lower bound']:bounds['upper bound']] = 0
    x_unwindowed = x_clipped.copy()
    for i,bounds in noise_df.iterrows():
        lb = bounds['lower bound']
        ub = bounds['upper bound']
        x_clip = x[lb:ub].copy()
        window_function = signal.windows.tukey(x_clip.shape[0],alpha)
        x_clip *= window_function
        x_clipped[lb:ub] = x_clip[:]

    #full_clip = pd.Series(full_clip_list).explode()
    #x_clipped[:] = full_clip[:]
    #print(type(full_clip))
        #print(full_clip[:])
    X_clipped = librosa.stft(x_clipped)
    psd,mu,sigma = compute_psd_data(X_clipped)
    return psd,x_clipped,X_clipped


# ## Feature Extraction

# ### Spectral Features

# Compute basic spectral features in the following functions.

# In[73]:


def spectral_descriptors(spectrum,freqs,ext=''):
    label_centr = 'centroid' + ext
    spec_desc = {}
    centr = np.sum(spectrum*freqs)/np.sum(spectrum)
    spec_desc[label_centr] = centr
    
    label_RMS = 'RMS' + ext
    spread = np.sqrt(  
        np.sum(((freqs-centr)**2)*spectrum) / np.sum(spectrum)  )
    spec_desc[label_RMS] = spread
    
    label_skew = 'skew' + ext
    skew = ( np.sum(((freqs-centr)**3)*spectrum) /
        (np.sum(spectrum)*(spread**3)) )
    spec_desc[label_skew] = skew
    
    label_kurto = 'kurtosis' + ext
    kurto = ( np.sum(((freqs-centr)**4)*spectrum) /
        (np.sum(spectrum)*(spread**4)) )
    spec_desc[label_kurto] = kurto
    
    label_entropy = 'entropy' + ext
    entropy = -np.sum(spectrum*np.log(spectrum))
    spec_desc[label_entropy] = entropy
    
    return spec_desc


def sig_peak_finder_internal(input_signal,freqs,primary_widths=[1,200],prominence_limit=0.1,n_peaks=3):
    peaks,peak_features= signal.find_peaks(input_signal,width=[1,200],prominence=prominence_limit)
    peak_features['indexes'] = peaks
    peak_features['frequency'] = freqs[peaks]
    df = pd.DataFrame(peak_features)
    top_peaks = df.sort_values(df.columns[0],ascending=False,axis=0).head(n_peaks).reset_index()
    feature_dict = {}
    iter_freq = []
    iter_prom = []
    iter_width = []
    iter_acc = []
    iter_index = []
    iter_wh = []
    iterable_features = {}
    
    for index,peak in top_peaks.iterrows():
        ipsd = peak['indexes']
        secondary_width = peak['widths']
        lower_limit = max([0,int(ipsd-secondary_width-1)])
        upper_limit = min([int(ipsd+secondary_width+2),len(input_signal)])
        sig_window = input_signal[lower_limit:upper_limit]
        if len(sig_window) == 0: continue
        peaks = signal.find_peaks_cwt(sig_window,widths=np.arange(1,int(secondary_width+1)),wavelet=signal.ricker)+lower_limit
        acc_check = pd.Series([(np.abs(peaks-ipsd))/secondary_width,1]).explode().values
        accuracy = 1 - min(acc_check)
        
        label_freq = 'peak%i_freqency'%index
        label_prom = 'peak%i_prominence'%index
        label_width = 'peak%i_width'%index
        label_agr = 'peak%i_agreement'%index
        label_index = 'peak%i_index'%index
        label_wh = 'peak%s_width_height'%index
        
        feature_dict[label_freq] = peak['frequency']
        feature_dict[label_prom] = peak['prominences']
        feature_dict[label_width] = secondary_width
        feature_dict[label_agr] = accuracy
        #feature_dict[label_index] = int(ipsd)
        feature_dict[label_wh] = peak['width_heights']
        
        iter_freq.append(peak['frequency'])
        iter_prom.append(peak['prominences'])
        iter_width.append(secondary_width)
        iter_acc.append(accuracy)
        iter_index.append(int(ipsd))
        iter_wh.append(peak['width_heights'])
        
    iterable_features['frequencies'] = iter_freq
    iterable_features['prominences'] = iter_prom
    iterable_features['widths'] = iter_width
    iterable_features['accuracies'] = iter_acc
    iterable_features['indexes'] = iter_index
    iterable_features['width_heights'] = iter_wh
        
    return feature_dict,iterable_features


def sig_peak_finder(input_signal,freqs,primary_widths=[1,200],prominence_limit=0.1,n_peaks=5,label_prefix='peak',extend_label=False,extension='_autocorr'):
    peaks,peak_features= signal.find_peaks(input_signal,width=[1,200],prominence=prominence_limit)
    peak_features['indexes'] = peaks
    peak_features['frequency'] = freqs[peaks]
    df = pd.DataFrame(peak_features)
    top_peaks = df.sort_values(df.columns[0],ascending=False,axis=0).head(n_peaks).reset_index()
    feature_dict = {}
    
    for index,peak in top_peaks.iterrows():
        ipsd = peak['indexes']
        secondary_width = peak['widths']
        lower_limit = max([0,int(ipsd-secondary_width)])
        upper_limit = min([int(ipsd+secondary_width+1),len(input_signal)])
        sig_window = psd[lower_limit:upper_limit]/input_signal_var
        if len(sig_window) == 0: continue
        peaks = signal.find_peaks_cwt(sig_window,widths=np.arange(1,int(secondary_width+1)),wavelet=signal.ricker)+lower_limit
        acc_check = pd.Series([(np.abs(peaks-ipsd))/secondary_width,1]).explode().values
        accuracy = 1 - min(acc_check)
        
        s_index = label_prefix+str(index)
        if extend_label: s_index+=extension
        
        label_freq = '%s_freqency'%s_index
        label_prom = '%s_prominence'%s_index
        label_width = '%s_width'%s_index
        label_acc = '%s_agreement'%s_index
        label_index = '%s_index'%s_index
        label_wh = '%s_width_height'%s_index
        
        feature_dict[label_freq] = peak['frequency']
        feature_dict[label_prom] = peak['prominences']
        feature_dict[label_width] = secondary_width
        feature_dict[label_acc] = accuracy
        #feature_dict[label_index] = int(ipsd)
        feature_dict[label_wh] = peak['width_heights']
        
    return feature_dict


# Features of interest for long audio include:
# * `primary_label`
# * geo-spatial data: `longitude`, `latitude`, and `date`
# * clip window `seconds` in `train_soundscape_labels`
# * spectral features
# 
# The test data are labeled by date and location, so it is best to parse info from the files names in `train_soundscape` for labeling instead of going directly to the metadata. The following function parses a file and returns the relevant metadata.

# In[ ]:


def long_audio_metadata(file_path,long_location_path=long_loc_path,ext=default_audio_extension[0]):
    metadata_dict = {}
    filename_index = file_path.rfind('/')
    filename = file_path[filename_index+1:]
    logging.info('Filename recovered: %s'%filename)
    metadata_dict['filename'] = filename
    
    file_id,location,date = filename.split('_')
    date = date[:-4]
    date = parse(date).date()
    logging.info('id: %s, loc: %s, date: %s'%(file_id,location,date))
    metadata_dict['date'] = date
    metadata_dict['audio_id'] = file_id
    metadata_dict['site'] = location
    location_file_ext = '_recording_location.txt'
    location_file = long_location_path + location + location_file_ext
    logging.info('location file path: %s'%location_file)
    with open(location_file,'r') as f:
        lines = f.readlines()
        lat  = float(str(lines[3]).split(':')[1])
        long = float(str(lines[4]).split(':')[1])
        logging.info('%s: latitude %f, longitude %f'%(location,lat,long))
    metadata_dict['longitude'] = long
    metadata_dict['latitude']  = lat
    return metadata_dict
        
def short_audio_metadata(file_path,metadata_path=metadata_path):
    columns = ['primary_label','secondary_labels','latitude','longitude','date','filename']
    metadata = pd.read_csv(metadata_path)
    filename_index = file_path.rfind('/')
    filename = file_path[filename_index+1:]
    metadata = metadata.loc[metadata['filename']==filename]
    metadata = metadata[columns]
    return metadata.to_dict(orient='records')[0]

def extract_spectral_features_short(audio_path):
    t0 = time()
    #load short metadata
    export_dict = short_audio_metadata(audio_path)
    #logging.info('Beginning extraction of {} at {}'.format(
    #    export_dict['filename'].values,date.fromtimestamp(time())))
    
    # load and stft
    x,X,freqs,window_function = librosa_load_both(audio_path)
    # noise reduction 0th order
    x_NR2,X_NR2,freqs = noise_reduction(x,X,window_function)
    # find peaks for second noise reduction
    #peaks, peaks_df = find_x_peaks(x_NR)
    #psd,x_c,X_c = characterize_noise_psd(x_NR,peaks_df)
    # noise reduction 1st order
    #x_NR2,X_NR2,freqs = noise_reduction(x,X_c,window_function,filter_red=True,filter_red_val=128)
    #analyze denoised signal
    psd_NR2,mu,sigma = compute_psd_data(X_NR2)
    #psd_NR2,mu,sigma = compute_psd_data(X_NR)
    
    #extract features from final denoised signal
    features_dict,features_iter = sig_peak_finder_internal(psd_NR2,freqs,n_peaks=3)
    export_dict.update(features_dict)
    
    psd,psd_mu,psd_std = compute_psd_data(X_NR2)
    spectral_desc = spectral_descriptors(psd,freqs)
    export_dict.update(spectral_desc)
    
    Xcov_NR2 = np.cov(X_NR2)
    for index,peak_index in enumerate(features_iter['indexes']):
        cov_spectrum = Xcov_NR2[peak_index,:]
        cov_spec_desc = spectral_descriptors(cov_spectrum,freqs,ext='_crosscor%s'%str(index))
        export_dict.update(cov_spec_desc.copy())
    tf = time()
    logging.info('Finished processing %s'%audio_path)
    logging.info('Time elapsed: %f seconds'%(tf-t0))
        
    return export_dict

def extract_spectral_features_long(audio_path,sr=default_sample_rate,n_fft=default_n_fft_fine,                   train_metadata=False,train_metadata_path=long_metadata_path):
    if train_metadata: t_df = pd.read_csv(train_metadata_path)
    t0 = time()
    #load short metadata
    metadata_dict = long_audio_metadata(audio_path)
    #logging.info('Beginning extraction of {} at {}'.format(
    #    export_dict['filename'].values,date.fromtimestamp(time())))
    x_precut = librosa.load(audio_path,sr=default_sample_rate)
    
    export_dict_list = []
    chunk_length = 5*default_sample_rate
    elapsed = 0
    length = len(x_precut)
    
    while elapsed < length:
        export_dict = {}
        export_dict.update(metadata_dict.copy())
        if elapsed == 0:
            x_noise_estimator = x_precut[elapsed+chunk_length:elapsed+3*chunk_length].copy()
            x_signal_region   = x_precut[elapsed:elapsed+chunk_length].copy()
        elif elapsed+chunk_length > length:
            x_noise_estimator = x_precut[elapsed-2*chunk_length:elapsed].copy()
            x_signal_region   = x_precut[elapsed:length].copy()
        else:
            x_noise_estimator = x_precut[int(elapsed-1.5*chunk_length):int(elapsed+1.5*chunk_length)].copy()
            x_signal_region   = x_precut[elapsed:elapsed+chunk_length].copy()
        
        wf = signal.windows.tukey(x_noise_estimator.shape[0],default_alpha)
        
        X_noise_estimator = librosa.stft(wf*x_noise_estimator)
        freqs = np.arange(0, 1 + n_fft / 2) * sr / n_fft

        # load and stft
        # noise reduction 0th order
        x_NR,X_NR,freqs = noise_reduction(x_noise_estimator,X_noise_estimator,window_function)
        # find peaks for noise reduction
        peaks, peaks_df = find_x_peaks(x_NR)
        psd,x_c,X_c = characterize_noise_psd(x_NR,peaks_df)
        # noise reduction 1st order
        x_NR2,X_NR2,freqs = noise_reduction(x,X_c,window_function,filter_red=True,filter_red_val=128)
        #analyze denoised signal
        psd_NR2,mu,sigma = compute_psd_data(X_NR2)

    #extract features from final denoised signal
        features_dict,features_iter = sig_peak_finder_internal(psd_NR2,freqs,n_peaks=3)
        export_dict.update(features_dict)

        spectral_desc = spectral_descriptors(psd_NR2,freqs)
        export_dict.update(spectral_desc)

        Xcov_NR2 = np.cov(X_NR2)
        for index,peak_index in enumerate(features_iter['indexes']):
            cov_spectrum = Xcov_NR2[peak_index,:]
            cov_spec_desc = spectral_descriptors(cov_spectrum,freqs,ext='_crosscor%s'%str(index))
            export_dict.update(cov_spec_desc.copy())
            
        elapsed_seconds = int(elapsed/32000)+5
        row_id_str = '{}_{}_{}'.format(metadata_dict['audio_id'],metadata_dict['site'],elapsed_seconds)
        row_id_dict = {'row_id':row_id_str}
        row_id_dict.update({'seconds':elapsed_seconds.copy()})
        export_dict.update(row_id_dict)
        if train_metadata:
            birds_value = train_metadata.loc[train_metadata['seconds']==elapsed_seconds]['birds'].values
            birds_dict = {'birds':birds_value}
            export_dict.update(birds_dict)
        elapsed += chunk_length
        export_dict_list.append(export_dict.copy())
    tf = time()
    logging.info('Finished processing %s'%audio_path)
    logging.info('Time elapsed: %f seconds'%(tf-t0))

    return export_dict_list

def preprocess_short_audio(range_specify=False,range_upper=397,range_lower=0):
    if not os.path.exists(save_path): os.mkdir(save_path)
    bird_list = glob(short_audio_path+'*')
    if range_specify:
        bird_list = bird_list[range_lower:range_upper]
    
    for bird_name in bird_list:
        df_name = bird_name + '_preprocess.csv'
        df = pd.DataFrame(data=None)
        short_audio = load_audio_files(bird_name,ext=default_audio_extension)
        for audio_file in short_audio:
            export_dict = extract_spectral_features_short(audio_file)
            df = df.append(export_dict,ignore_index=True)
        df.to_csv(df_name)
        
def preprocess_long_audio(range_specify=False,range_upper=20,range_lower=0):
    if not os.path.exists(save_path): os.mkdir(save_path)
    bird_list = glob(long_audio_path+'*')
    if range_specify:
        bird_list = bird_list[range_lower:range_upper]
    
    for bird_name in bird_list:
        df_name = bird_name[:-4] + '_preprocess_long.csv'
        df = pd.DataFrame(data=None)
        short_audio = load_audio_files(short_audio_path,ext=default_audio_extension)
        for audio_file in short_audio:
            export_dict_list = extract_spectral_features_short(audio_file)
            df = df.append(export_dict_list,ignore_index=True)
        df.to_csv(df_name)
        


# # References
# 
# * [1] Harris F J 1978 Proceedings of the IEEE6651–83 ISSN 0018-921 
# * [2] B P Abbott et al 2020 Class. Quantum Grav. 37 055002, arXiv:1908.11170 [gr-qc]

# In[ ]:




