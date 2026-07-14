#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:11:18 2025

@author: zixuanwang

ACF
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf

from _paths import find_input

onp.random.seed(10)

#################################################
# 1. Divide data y into S shards y_1...y_s.
#################################################

# 1.1 define data and model
data1 = onp.loadtxt(find_input("SimARTFIMA11.txt"))
data2 = pd.read_csv(find_input("Vancouver_AR2_TFI_MA1.csv"))["y"].to_numpy()

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
