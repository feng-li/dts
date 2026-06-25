#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:11:18 2025

@author: zixuanwang

ACF
"""

import statsmodels.api as sm
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import numpy as onp
from jax import grad, hessian, jacobian
from numdifftools import Hessian as Hess_finite_diff
from numpy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, Bounds, basinhopping
import jax.scipy.stats as sps_jax
import progressbar
import pandas as pd
import pickle
import sys, os, platform
import warnings
from jax.numpy.linalg import inv, slogdet
from statsmodels.tsa.stattools import acf, adfuller


gtol = 1e-4 
max_iter_optim = 500 
onp.random.seed(10)

#################################################
# 1. Divide data y into S shards y_1...y_s.
#################################################

# 1.1 define data and model

if platform.system() == 'Darwin':
    proj_path = '/Users/' + os.getenv("USER") + '/OneDrive - UTS/Project 1/'
elif platform.system() == 'Windows':
    proj_path = '/Users/' + os.getlogin() + '/OneDrive - UTS/Project 1/'
else:
    raise ValueError()      


data2 = onp.load(proj_path + 'Datasets/Vancouver_AR2_TFI_MA1.npy')
data1 = onp.loadtxt(proj_path + 'Datasets/SimARTFIMA11_short.txt')

# Compute ACFs
acf_data1 = acf(data1, nlags=100)
acf_data2 = acf(data2, nlags=100)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot ACF for data1
axes[0].stem(range(len(acf_data1)), acf_data1)
axes[0].set_title('Simulated ARTFIMA (short memory)')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')

# Plot ACF for data2
axes[1].stem(range(len(acf_data2)), acf_data2)
axes[1].set_title('Vancouver Temperature (long memory)')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('ACF')

# Add overall title and adjust spacing
#fig.suptitle('ACF Comparison of Simulated and Real Data', fontsize=14, y=1.02)  # slightly above plots

plt.tight_layout()
plt.subplots_adjust(top=0.75)  # lower the top space to bring title closer
plt.show()
