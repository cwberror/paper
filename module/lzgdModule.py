from torch import nn
from torch.nn import Conv2d


class Lzgd(nn.Module):
    def __init__(self,out_channels):
        super(Lzgd,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),#批归一化处理(Batch Normalization, BN层)通常用于深层的神经网络中，其作用是对网络中某层特征进行标准化处理，其目的是解决深层神经网络中的数值不稳定的问题，使得同批次的各个特征分布相近，网络更加容易训练。
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=out_channels,out_channels=3,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(in_channels=3,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=out_channels,out_channels=3,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 )

    def forward(self,x):
        x=self.conv1(x)
        return x