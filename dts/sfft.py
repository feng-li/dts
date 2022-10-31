from systemds.context import SystemDSContext
import numpy as np
import pandas as pd

def dft_systemds(signal,name):


    with SystemDSContext(spark) as sds:
        size = signal.count()
        signal = sds.from_numpy(signal.toPandas().to_numpy())
        pi = sds.scalar(3.141592654)

        n = sds.seq(0,size-1)
        k = sds.seq(0,size-1)

        M = (n @ (k.t())) * (2*pi/size)

        Xa = M.cos() @ signal
        Xb = M.sin() @ signal

        index = (list(map(lambda x: [x],np.array(range(0, size, 1)))))
        DFT = np.hstack((index,Xa.cbind(Xb).compute()))
        DFT_pdf = pd.DataFrame(DFT, columns=list(["id",name+'_sin',name+'_cos']))
        DFT_df = spark.createDataFrame(DFT_pdf)
        return DFT_df
