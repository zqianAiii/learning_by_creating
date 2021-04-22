import torch
import torch.nn as nn
import torch.nn.functional as F


#DARTS operations
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}


class ReLUConvBN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine = True):
        super(ReLUConvBN, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = False),
            nn.GroupNorm(1, out_channels, affine = affine)
        )
    
    def forward(self, x):
        return self.net(x)
    
    

class SepConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine = True):
        super(SepConv, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride = stride, padding = padding, groups = in_channels, bias = False),
            nn.Conv2d(in_channels, in_channels, 1, bias = False),
            nn.GroupNorm(1, in_channels, affine = affine),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding = padding, groups = in_channels, bias = False),
            nn.Conv2d(in_channels, out_channels, 1, bias = False),
            nn.GroupNorm(1, out_channels, affine = affine)
        )
    
    def forward(self, x):
        return self.net(x)
    
    
    
class DilConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, affine = True):
        super(DilConv, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride = stride, padding = padding, 
                      dilation = dilation, groups = in_channels, bias = False),
            nn.Conv2d(in_channels, out_channels, 1, bias = False),
            nn.GroupNorm(1, out_channels, affine = affine)
        )
        
    def forward(self, x):
        return self.net(x)
    

    
class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)
    
    
    
class FactorizedReduce(nn.Module):
    
    def __init__(self, in_channels, out_channels, affine = True):
        super(FactorizedReduce, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride = 2, bias = False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 1, stride = 2, bias = False)
        self.bn = nn.GroupNorm(1, out_channels, affine = affine)
        
    def forward(self, x):
        x = F.relu(x)
        return self.bn(torch.cat([self.conv1(x), self.conv2(x)], dim= 1))
            