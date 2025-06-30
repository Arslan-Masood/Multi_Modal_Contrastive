#!/usr/bin/env python
# Save this as measure_parquet_read.py

import pandas as pd
import time
from memory_profiler import memory_usage

def read_parquet():
    data_path = "/scratch/work/masooda1/datasets/Multi_Modal_Contrastive/centered.filtered.parquet"
    df = pd.read_parquet(data_path, engine='pyarrow')
    return df

if __name__ == "__main__":
    start_time = time.time()
    mem_usage = memory_usage((read_parquet, (), {}), interval=0.1, timeout=None, max_iterations=1)
    end_time = time.time()

    elapsed_time = end_time - start_time
    max_memory = max(mem_usage)

    print(f"Time taken to read the Parquet file: {elapsed_time:.2f} seconds")
    print(f"Peak memory usage: {max_memory:.2f} MiB")