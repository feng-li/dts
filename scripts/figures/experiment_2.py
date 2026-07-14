#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:02:48 2025

@author: zixuanwang
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

from _paths import find_input

# Load your DataFrame
df = pd.read_pickle(find_input("ARMA.pkl"))

# Specify your order, colours and line-styles
order       = ["MCMC", "G = 10", "G = 100", "G = 1000"]
palette     = {"MCMC":"tab:blue", "G = 10":"tab:orange", "G = 100":"tab:green", "G = 1000":"tab:red"}

# Draw the contours (no legend)
plt.figure(figsize=(8,6))
ax = sns.kdeplot(
    data=df,
    x="phi1", y="theta1",
    hue="Number of Groups",
    hue_order=order,
    palette=palette,
    levels=5,
    common_norm=False,
    fill=False,
    linewidths=3.2,
    bw_adjust=1.7,
    alpha=1,
    legend=False   # turn off Seaborn’s own legend
)


# Manual legend handles (same as before)
handles = [
    Line2D([], [], color=palette[name], 
           linewidth=3.2, label=name)
    for name in order
]

# Bigger legend text & title
ax.legend(
    handles=handles,
    title="Number of Groups",
    loc="upper right",
    fontsize=12,          # legend label size
    title_fontsize=14,   # legend title size
    handlelength=2,
    labelspacing=0.6,
    borderpad=0.5
)

# Then labels/title.
ax.set_xlabel(r"$\phi_1$")
ax.set_ylabel(r"$\vartheta_1$")
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.tick_params(axis='both', which='major', labelsize=14)
#ax.set_title("Posterior Approximations for ARMA(1,1), n=50,000", pad=6)

plt.tight_layout()
plt.show()
