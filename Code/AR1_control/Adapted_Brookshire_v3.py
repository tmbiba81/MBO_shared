#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:17:36 2023

@author: thomasbiba
"""

"""
Tests whether observed average signal has greater spectral power than 
surragate distributions matched for AR1 structure.
This adaptation of the methods proposed by Brooksire (2022) matches the 
AR1 strucutre to each underlying participant, as proposed by Landau et al. (2022).
Note that signficant effects depends on both (1) oscilations within participants and 
(2) phase synchrony accross participants

@author: katherineduncan 
"""

# %%import packages
import pathlib as Path
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# %%
# check if pathlib is imported
try:
    from pathlib import Path
except ImportError:
    print("pathlib is not available. Please install it.")
    sys.exit(1)

# specify the github directory 
git_dir = Path("/Users/thomasbiba/Documents/GitHub/MBO_shared")
type(git_dir)

#%% define functions

# discrete Forrier transform
def dft(x, fs, nfft, axis=0):

    f = np.fft.fftfreq(nfft, 1 / fs)
    y = np.abs(np.fft.fft(x, nfft, axis=axis))**2 #get power (square of amplitude)
    # Only keep positive frequencies
    f_keep = f > 0
    f = f[f_keep]
    y = y[f_keep]
    return f, y


# function to get AR1 simulated values for each participant
def ARMA_fit(ts, idx, mdl_order, nsim_exp, n):
    #model and get coefs (for one participant)
    mdl_ARMA = sm.tsa.ARMA(ts.to_numpy()[idx,], mdl_order).fit(trend='c', disp=0)            
    arma_process_ARMA = sm.tsa.ArmaProcess.from_coeffs(mdl_ARMA.arparams)
    #use coefs to simulate many timeseries (for one participant)
    x_sim_ARMA = np.empty([nsim_exp,n+2])
    for sim in range(nsim_exp):
        x_sim_ARMA[sim,] = np.append(arma_process_ARMA.generate_sample((1,n),
                                            scale=mdl_ARMA.resid.std(), axis=1),[sim,idx]) #append sim and subj number to output
    return(x_sim_ARMA)  # nsim time series of AR1 stimulated experiments - for one participant (array)


# %% define main function 

def ARMA_run(dataPath):

    # select the dependent measure from the file name
    filename = os.path.basename(dataPath)         # 'example_data_file.csv'
    base_name = os.path.splitext(filename)[0]      # 'example_data_file'
    dv_name = base_name.split('_')[1]
    
    # set seed before running 
    np.random.seed(42)

    # parameters for computing features for each signal
    nsim_exp=3000 #100 # sim for each experiment  # change to 3000 (per participant)
    fs = 30  # sampling frequency in hz
    freq_cutoff = [3,14]   # frequencies to look at

    # ARMA1 paramaters
    mdl_order = (1, 0)  # style of autocorrelation we are captureing

    # load data
    sig_raw = pd.read_csv(dataPath)  # read in data file of raw time-series
    nSub = len(sig_raw)

    # drop subject number column
    sig_raw.drop(sig_raw.columns[0],axis=1,inplace=True)
    n = len(sig_raw.iloc[0,:])


    # simulate noise matched to ARMA1 strucutre of each participant 
    ARMA_cat = pd.DataFrame()
    for subi in range(nSub):
        ARMA_s = ARMA_fit(sig_raw, subi, mdl_order, nsim_exp, n)
        ARMA_cat = pd.concat([ARMA_cat,pd.DataFrame(ARMA_s)]).reset_index(drop=True)
        print(subi)

    ARMA_cat.rename(columns={ARMA_cat.columns[n]: 'sim_num', ARMA_cat.columns[n+1]: 'sub_num'},inplace=True)

    # normalize and detrend each participant's time series
    sig_z = sp.zscore(sig_raw,1)
    sig_detrend = sm.tsa.tsatools.detrend(sig_z, order=2, axis=1)  # detrend linear and quadratic trends

    # get mean and filter
    sig_mean = np.mean(sig_detrend, axis=0)
    hanW = np.hanning(len(sig_mean))
    sig_mean = sig_mean*hanW
    sig_zpad = np.append(sig_mean, np.zeros(30-len(sig_mean))) #zero padd to 30 to get even frequency estimates
    
    # run fft on observed data
    nfft = len(sig_zpad)
    f_sig, y_sig = dft(sig_zpad, fs, nfft)

    # normalize and detrend each suragate participant's time series 
    AR1_z = sp.zscore(sp.zscore(ARMA_cat.iloc[:,0:n]),1) #note that we need to avoid participant number and simulation columns
    AR1_detrend = pd.DataFrame(sm.tsa.tsatools.detrend(AR1_z, order=2, axis=0))  # detrend: converting to dataframe b/c outputed as ndarray. This depdends on somthing about the data structure
    AR1_detrend['sim_num'] = ARMA_cat['sim_num'] #add simulation column back in

    # get mean and filter suragate participants within each simulation
    AR1_means = AR1_detrend.groupby(['sim_num']).mean()  # getting the mean across subject per simulation 
    AR1_means = AR1_means * hanW
    AR1_zpad= np.append(AR1_means, np.zeros([nsim_exp,30-len(sig_mean)]),1) #zero padd to 30 to get even frequency estimates

    # run fft on suragate data
    y_sim = np.empty([nsim_exp,len(f_sig)])
    for sim in range(nsim_exp):
        _, y_sim[sim,] = dft(AR1_zpad[sim,:], fs, nfft)

    # get non-parametric statistics
    z = (y_sig-np.mean(y_sim,0))/np.std(y_sim,0)
    p_raw = 1-sp.norm.cdf(z)

    # Select the frequency range
    freq_sel = (freq_cutoff[0] <= f_sig) * (f_sig <= freq_cutoff[1])
    f = f_sig[freq_sel]
    y_sig = y_sig[freq_sel]
    y_sim = y_sim[:,freq_sel]
    p_raw = p_raw[freq_sel]
    z = z[freq_sel]

    _, p_corr, _, _ = multipletests(p_raw, method='fdr_bh')

    # create output dictionary
    out = {'dv': dv_name,
           'p_raw': p_raw, 
           'p_corr':p_corr, 
           'y_sig': y_sig,
           'y_95': np.percentile(y_sim, 95, axis=0), 
           'freq': f,
           'z': z}
    
    # plot results
    plt.plot(f, y_sig, label='Spectrum')
    plt.plot(f,np.percentile(y_sim, 95, axis=0),
        '--k', label='95% CI')
    signif = p_corr < 0.05
    plt.plot(f[signif], y_sig[signif],
        '*r', label='p < .05')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('power')
    plt.xlim(freq_cutoff[0], freq_cutoff[1])
    plt.legend()  

    return out 

# %%
ts_path = git_dir / str("Data") / str("Timeseries")
files = os.listdir(ts_path)
# List files with a specific extension (e.g., .txt)
csv_file_paths = [file for file in ts_path.iterdir() if file.is_file() and file.suffix == ".csv"]
print("Text files:", csv_file_paths)
# extract file names from the list of paths
files = [fp.name for fp in csv_file_paths]
# Remove .csv extension from each filename in files
file_substrings = [f[:-4] if f.endswith('.csv') else f for f in files]
print("File names:", file_substrings)
# add these names to the list
out=list()

# %% 
for fp in csv_file_paths:
    #dataPath= os.path.join(path_base, f)    
    out.append(ARMA_run(fp))

# %% prep output for csv files
output_path = git_dir / str("Data") / str("AR1_control")
df_out = pd.DataFrame()
for f in range(3):
    mat_out = np.vstack((out[f]['freq'], out[f]['y_sig'], out[f]['y_95'], 
    out[f]['z'], out[f]['p_raw'], out[f]['p_corr'])).transpose()
    mat_out = pd.DataFrame(mat_out, columns=['freq', 'power', 'surragate_95', 'z_stat', 'p_raw', 'p_corr'])
    mat_out['version'] = files[f][-6:-4]
    mat_out['dv'] = out[f]['dv']
    df_out = pd.concat((df_out, mat_out))

df_out.to_csv(os.path.join(output_path, 'adapted_brooksire_2024_TB6.csv'))

# print "saved to {output_path}"
print(f"adapted_brooksire_2024_TB6.csv saved to {output_path}")



# %%
