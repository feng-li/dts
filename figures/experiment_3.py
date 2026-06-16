#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:02:48 2025

@author: zixuanwang
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1) Load your new DataFrame
df = pd.read_pickle(os.path.join(os.getcwd(), "ARTFIMA.pkl"))
print(df.head())

# 2) Specify the three partitions, and choose distinct colors & line-styles
order       = ["MCMC", "Systematic", "Sequential"]
palette     = {
    "MCMC":       "tab:blue",
    "Systematic": "tab:orange",
    "Sequential": "tab:green"
}


# 3) Draw the overlaid KDE contours (legend turned off)
plt.figure(figsize=(8, 6))
ax = sns.kdeplot(
    data=df,
    x="phi1",
    y="theta1",
    hue="Partition",
    hue_order=order,
    palette=palette,
    levels=5,
    common_norm=False,
    fill=False,
    linewidths=2.2,
    bw_adjust=3.5,
    alpha=1,
    legend=False
)

# 4) Build manual legend handles that match your plot
handles = [
    Line2D([], [],
           color=palette[name],
           linewidth=2.2,
           alpha=1,
           label=name)
    for name in order
]

# 5) Add the legend, bumping up font sizes
ax.legend(
    handles=handles,
    title="Partition",
    loc="upper right",
    fontsize=12,
    title_fontsize=14,
    handlelength=2.5,
    labelspacing=0.5,
    borderpad=0.4
)

# 6) Axis labels & tick numbers larger
ax.set_xlabel(r"$\phi_1$", fontsize=16)
ax.set_ylabel(r"$\vartheta_1$", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)

# 7) Add a little margin around your contours
phi_min, phi_max     = df["phi1"].min(), df["phi1"].max()
theta_min, theta_max = df["theta1"].min(), df["theta1"].max()
dx = (phi_max - phi_min) * 0.05
dy = (theta_max - theta_min) * 0.05
ax.set_xlim(phi_min - dx, phi_max + dx)
ax.set_ylim(theta_min - dy, theta_max + dy)

plt.tight_layout()
plt.show()