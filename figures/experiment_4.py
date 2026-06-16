#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:02:48 2025

@author: zixuanwang
"""
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1) Load your new DataFrame
df = pd.read_pickle(os.path.join(os.getcwd(), "Vanc_AR1.pkl"))
print(df.head())

# 2) Specify the three partitions, and choose distinct colors & line-styles
order       = ["MCMC", "Frequency Domain", "Time Domain"]
palette     = {
    "MCMC":             "tab:blue",
    "Frequency Domain": "tab:orange",
    "Time Domain":      "tab:green"
}


# Make a copy with transformed sigma2
df_plot = df.copy()
df_plot["sigma2"] = np.exp(df_plot["log_sigma2"])

# 3) Draw the overlaid KDE contours
plt.figure(figsize=(8, 6))
ax = sns.kdeplot(
    data=df_plot,
    x="phi1",
    y="sigma2",               # <-- changed here
    hue="Strategy",
    hue_order=order,
    palette=palette,
    levels=5,
    common_norm=False,
    fill=False,
    linewidths=2.2,
    bw_adjust=1.5,
    alpha=1,
    legend=False
)

# 4) Manual legend
handles = [
    Line2D([], [], color=palette[name], linewidth=2.2, alpha=1, label=name)
    for name in order
]

ax.legend(
    handles=handles,
    title="Strategy",
    loc="upper right",
    fontsize=12,
    title_fontsize=14,
    handlelength=2.5,
    labelspacing=0.5,
    borderpad=0.4
)

# 5) Axis labels
ax.set_xlabel(r"$\phi_1$", fontsize=16)
ax.set_ylabel(r"$\sigma^2$", fontsize=16)   # <-- updated label
ax.tick_params(axis="both", which="major", labelsize=14)

# 6) Add margin
phi_min, phi_max     = df_plot["phi1"].min(), df_plot["phi1"].max()
sigma_min, sigma_max = df_plot["sigma2"].min(), df_plot["sigma2"].max()
dx = (phi_max - phi_min) * 0.05
dy = (sigma_max - sigma_min) * 0.05
ax.set_xlim(phi_min - dx, phi_max + dx + 0.002)
ax.set_ylim(sigma_min - dy, sigma_max + dy)

plt.tight_layout()
plt.show()