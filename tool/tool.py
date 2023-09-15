import numpy as np
import pandas as pd
import torch


def getValueCount(tensor):
    ndarray=tensor2numpy(tensor)

    unique, count = np.unique(ndarray, return_counts=True)
    data_count = dict(zip(unique, count))
    df = pd.DataFrame(data_count, index=[0]).T.reset_index()
    df=df.rename(columns={'index':'数值',0:'数量'})
    df.to_csv('tensor数值统计.csv',index=False)
    return data_count


def tensor2numpy(tensor):
    ndarray=None
    if isinstance(tensor,torch.Tensor):
        if tensor.data.is_cuda:
            tensor = tensor.cpu().detach()
        ndarray=tensor.numpy()
    else:
        ndarray=tensor
    return ndarray


def Dim3to4(tensor):
    if len(tensor.shape)==3:
        return tensor.unsqueeze(dim=0)
    else:
        return tensor

def Dim4to3(tensor):
    if len(tensor.shape)==4:
        return tensor.squeeze()
    else:
        return tensor