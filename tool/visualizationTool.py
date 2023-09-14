import torch
from torch import nn

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import tool

import matplotlib.pyplot as plt

def showImageByElementOfDataset(data,style=None):
    img,label=data
    img=img.squeeze()
    if img.shape[0]==3:
        img=img.permute(1,2,0)
        plt.imshow(img, cmap=style)
    else:
        plt.imshow(img,cmap=style)

def showImage(tensor,style=None):
    tensor=tool.tensor2numpy(tensor)
    tensor=tensor.squeeze()
    if tensor.shape[0]==3:
        tensor=tensor.permute(1,2,0)
        plt.imshow(tensor, cmap=style)
    else:
        plt.imshow(tensor,cmap=style)
