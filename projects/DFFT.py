#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
import cmath

def distributed_dft(data, sc, num_slices):
    """
    A fully distributed DFT (not Cooley–Tukey) via broadcast + mapPartitions. This is N^2 compelaxity
    data: list or array of length N
    sc: SparkContext
    num_slices: how many partitions to split the work across
    Returns: list of length N, the DFT X[0..N-1]
    """
    N = len(data)
    # 1) broadcast the entire input
    x_b = sc.broadcast(data)

    def compute_chunk(ks):
        x = x_b.value
        N = len(x)
        out = []
        for k in ks:
            s = 0+0j
            for n, xn in enumerate(x):
                s += xn * cmath.exp(-2j * cmath.pi * k * n / N)
            out.append((k, s))
        return out

    # 2) parallelize the k‐indices
    ks_rdd = sc.parallelize(range(N), numSlices=num_slices)

    # 3) in each partition, compute X[k] for its ks
    Xrdd = ks_rdd.mapPartitions(lambda it: compute_chunk(list(it)))

    # 4) collect & sort by k
    return (
        Xrdd
        .sortByKey()
        .map(lambda kv: kv[1])
        .collect()
    )

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DistributedDFT").getOrCreate()
    sc = spark.sparkContext

    # example
    import numpy as np
    data = np.random.random(16).tolist()
    # use, say, 4 partitions for the 16 outputs
    X_spark = distributed_dft(data, sc, num_slices=4)
    X_np    = np.fft.fft(data).tolist()

    print("Spark DFT:", X_spark)
    print("NumPy FFT:", X_np)

    spark.stop()


from pyspark.sql import SparkSession
import numpy as np, math, cmath

def compute_subfft(kv, P, M):
    p, items = kv
    # sort by r = (global_idx - p)//P
    items = sorted(items, key=lambda x: (x[0] - p)//P)
    vals = [v for _, v in items]            # length M
    fft_vals = np.fft.fft(vals)             # M-point DFT
    # emit (k = r + p*M, Y_p[r])
    return [(r + p*M, fft_vals[r]) for r in range(M)]

def DFFT(rdd, N):
    P = rdd.getNumPartitions()
    assert (P & (P-1)) == 0 and N % P == 0    
    M = N // P
    # 1) do the M-point FFT on each of the P interleaved slices
    subffts = (
        rdd
        .zipWithIndex()                                        # (value, idx)
        .map(lambda vi: (vi[1] % P, (vi[1], vi[0])))            # (p, (idx, val))
        .groupByKey(numPartitions=P)
        .flatMap(lambda kv: compute_subfft(kv, P, M))          # (k, Y_p[r])
        .cache()
    )

    # 2) post-twiddle by W_N^(p·r)
    twiddled = subffts.map(lambda kv: (
        kv[0],
        kv[1] * cmath.exp(-2j * math.pi * (kv[0]//M)*(kv[0]%M) / N)
    ))

    # 3) now for each r (k % M), gather the P values and do a length-P FFT in Python
    final = (
        twiddled
        .map(lambda kv: (kv[0] % M, (kv[0]//M, kv[1])))  # → (r, (p, val))
        .groupByKey(numPartitions=M)
        .flatMap(lambda kv: [
            # kv[0] is r, kv[1] is iterable of (p, val) pairs
            (q*M + kv[0], fft_val)
            for q, fft_val in enumerate(
                np.fft.fft([v for _, v in sorted(kv[1])])
            )
        ])
        .sortByKey()
        .map(lambda kv: kv[1])
    )
    return final

if __name__ == "__main__":
    spark = SparkSession.builder.appName("FFT-GroupByR").getOrCreate()
    sc = spark.sparkContext

    # sanity-check for P=1,2,4,8,16, 
    # N: length of the series
    # P: number of partitions
    # P should be a power of two
    # P divides N evenly
    
    N = 160
    for P in (1,2,4,8,16):
        if N % P: continue
        data = np.random.random(N).tolist()
        rdd  = sc.parallelize(data, numSlices=P)
        out  = DFFT(rdd, N).collect()
        print(f"P={P} → match? {np.allclose(out, np.fft.fft(data))}")

    spark.stop()


# In[ ]:




