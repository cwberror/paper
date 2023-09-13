import numpy as np
import pandas as pd

def getValueCount(tensor):
    unique, count = np.unique(tensor.numpy(), return_counts=True)
    data_count = dict(zip(unique, count))
    df = pd.DataFrame(data_count, index=[0]).T
    return data_count
