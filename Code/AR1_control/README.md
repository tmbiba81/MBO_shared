# AR1 Control Analysis 

This code runs FFT analysis on memory SOA timeseries, with surrogate time-series generated with AR1 (autoregressive) noise, to evalaute if rythmic memory formation oscillates more than would be extected even if AR1 noise was present in the surrogate data. 

## Setup

### 1. Install Conda

If you don't have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) for macOS.

### 2. Create a Conda Environment

Open your terminal and run:

```bash
conda create -n AR1-control python=3.8

conda activate AR1-control

conda install numpy pandas scipy statsmodels matplotlib
```

### 3. Note

When running Adapted_Brookshire_v3.py, you may need to run each code chunk independently and sequentially for the code to run smoothly. 
