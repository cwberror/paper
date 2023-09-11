import torch.nn as nn
import torch
import torch.nn.functional as F
#源码网址：https://zhuanlan.zhihu.com/p/142985678

#两次卷积操作
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    '''
    解释下，上述的Pytorch代码：torch.nn.Sequential是一个时序容器，Modules 会以它们传入的顺序被添加到容器中。比如上述代码的操作顺序：卷积->BN->ReLU->卷积->BN->ReLU。

    DoubleConv模块的in_channels和out_channels可以灵活设定，以便扩展使用。
    
    如上图所示的网络，in_channels设为1，out_channels为64。
    
    输入图片大小为572*572，经过步长为1，padding为0的3*3卷积，得到570*570的feature map，再经过一次卷积得到568*568的feature map。
    
    计算公式：O=(H−F+2×P)/S+1
    
    H为输入feature map的大小，O为输出feature map的大小，F为卷积核的大小，P为padding的大小，S为步长。
'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


#下采样
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    '''
    这里的代码很简单，就是一个maxpool池化层，进行下采样，然后接一个DoubleConv模块。
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#上采样+特征融合
class Up(nn.Module):
    """Upscaling then double conv"""
    '''
    首先是__init__初始化函数里定义的上采样方法以及卷积采用DoubleConv。上采样，定义了两种方法：Upsample和ConvTranspose2d，也就是双线性插值和反卷积。
    
    在forward前向传播函数中，x1接收的是上采样的数据，x2接收的是特征融合的数据。特征融合方法就是，上文提到的，先对小的feature map进行padding，再进行concat。
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


#输出
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#构成unet网络
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        print(type(x), x.shape)
        logits = self.outc(x)
        return logits