import numpy as np
import torch
import multiprocessing.shared_memory as shared_mem
from multiprocessing import Pool, Array, Manager
import multiprocessing
import gc

core_num=multiprocessing.cpu_count()
chunk_size=10


def update_data(idx):
    #global clone_x
    for u in range(idx,min(idx+chunk_size,85)):
        clone_x[u]=4*clone_x[u]

if __name__=="__main__":
    clone_x=np.ones((10000,5),dtype=np.float)
    gc.collect()
    with Manager() as manager:
        past=clone_x[0][0]
        clone_x=manager.list(clone_x)
        with Pool(core_num) as p:
            p.map(update_data, range(0,10000,chunk_size))
        clone_x=np.array(clone_x)

        clone_x=torch.tensor(clone_x)
        cur=clone_x[0][0]
    print(clone_x[0])
    print(f"type(clone_x) : {type(clone_x)}")
    print(f"clone_x : {clone_x.dtype}")
    print(f"Is it changed? {past} {cur}")

    del clone_x
    gc.collect()
    