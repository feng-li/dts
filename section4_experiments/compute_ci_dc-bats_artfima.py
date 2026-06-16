'''
This code tries to compute 95%CI of samples from DC-BATS ARTFIMA code.
'''



import numpy as np

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

# -------- path --------
rds_path = "/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/DC-BATS/results/realdata_ARTFIMA_G10_full_and_shards.rds"

# -------- R readRDS --------
base = importr("base")
obj = base.readRDS(rds_path)   # R list: meta, full, shards

# Extract components
full_r = obj.rx2("full")       # matrix (or data.frame)
shards_r = obj.rx2("shards")   # list of matrices

# ---- full -> pandas -> numpy ----
with localconverter(ro.default_converter + pandas2ri.converter):
    full_pd = ro.conversion.rpy2py(full_r)
full_np = full_pd.to_numpy() if hasattr(full_pd, "to_numpy") else np.asarray(full_pd)

# ---- shards -> list[pandas] -> numpy stack ----
shards_np_list = []
G = len(shards_r)

for k in range(G):
    shard_k_r = shards_r.rx2(k + 1)  # R is 1-indexed
    with localconverter(ro.default_converter + pandas2ri.converter):
        shard_k_pd = ro.conversion.rpy2py(shard_k_r)
    shard_k_np = shard_k_pd.to_numpy() if hasattr(shard_k_pd, "to_numpy") else np.asarray(shard_k_pd)
    shards_np_list.append(shard_k_np)

shards_np = np.stack(shards_np_list, axis=0)

print("full_np shape:", full_np.shape)
print("shards_np shape:", shards_np.shape)  # (G, n_iter, n_params)

# 可选：直接保存成 npy / npz 给后续画图用
out_npz = "/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/DC-BATS/results/realdata_ARTFIMA_G10_full_and_shards.npz"
np.savez_compressed(out_npz, full=full_np, shards=shards_np)
print("saved:", out_npz)