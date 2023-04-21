import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModelClass import BaseModel

class VNet(BaseModel):

    def __init__(self, in_channels=1, nclass=1):
        super(VNet, self).__init__()


        self.in_tr = InputTransition(in_channels, 16)
        self.down_tr16 = DownTransition(16, 2)
        self.down_tr32 = DownTransition(32, 3)
        self.down_tr64 = DownTransition(64, 3)
        self.down_tr128 = DownTransition(128, 3)
        self.up_tr128 = UpTransition(256, 3)
        self.up_tr64 = UpTransition(128, 3)
        self.up_tr32= UpTransition(64, 2)
        self.up_tr16= UpTransition(32, 1)
        self.out_tr = OutputTransition(16, nclass)
        self.test_cell = TestCell(nclass)

    def forward(self, x):
        out8 = self.in_tr(x)
        out16 = self.down_tr16(out8)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.up_tr16(out, out8)
        out = self.out_tr(out)
        return out

class TestCell(nn.Module):
    def __init__(self,chans):
        super(TestCell, self).__init__()
        self.conv1 = nn.Conv3d(chans, 8 * chans, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv3d(8 * chans, chans, kernel_size = 3, padding = 1)
    def forward(self, input):
        return self.conv2(self.conv1(input))

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, 
                            self.weight, self.bias, True, self.momentum, self.eps)
    
class InputTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inChans, outChans, kernel_size = 3, padding = 1),
            ContBatchNorm3d(outChans),
            )
        self.relu =  nn.PReLU(outChans)

    def forward(self, x):
        out = self.conv1(x)
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        out = self.relu(torch.add(out, x16))
        return out 

class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        self.outconv = nn.Sequential(
            nn.Conv3d(inChans, 8, kernel_size = 3, padding = 1),
            ContBatchNorm3d(8),
            nn.PReLU(8),
            nn.Conv3d(8, outChans, kernel_size = 3, padding = 1),
            ContBatchNorm3d(outChans)
            ) 

    def forward(self, x):
        out = self.outconv(x)
        return out 

class DownTransition(nn.Module):
    def __init__(self, inChans, nConv):
        super(DownTransition, self).__init__()
        self.downconv = nn.Sequential(
            nn.Conv3d(inChans, 2 * inChans, kernel_size = 2, stride = 2),
            ContBatchNorm3d(2 * inChans),
            )
        self.conv = _make_nLUConvBN(2 * inChans, nConv)
        self.relu = nn.PReLU(2 * inChans)

    def forward(self, x):
        x = self.downconv(x)
        convx = self.conv(x)
        out = self.relu(torch.add(x, convx))
        return out
        
class UpTransition(nn.Module):
    def __init__(self, inChans, nConv):
        super(UpTransition, self).__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(inChans, inChans // 2, kernel_size = 2, stride = 2),
            ContBatchNorm3d(inChans // 2),
            )
        self.conv = _make_nLUConvBN(inChans // 2, nConv)
        self.relu = nn.PReLU(inChans // 2)

    def diffpad(self, out ,diff):
        diffZ = diff.size()[2] - out.size()[2]
        diffX = diff.size()[3] - out.size()[3]
        diffY = diff.size()[4] - out.size()[4]
        out = F.pad(out, [diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2,
                        diffZ//2, diffZ - diffZ //2])
        return out

    def forward(self, x, skipx):
        x = self.upconv(x)
        x = self.diffpad(x, skipx)
        x = torch.add(skipx, x)
        out = self.relu(self.conv(x))
        return out

class LUConvBN(nn.Module):
    def __init__(self, nchan):
        super(LUConvBN, self).__init__()
        self.relu1 = nn.PReLU(nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size = 3, padding = 1)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.bn1(self.conv1(self.relu1(x)))
        return out

def _make_nLUConvBN(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConvBN(nchan))
    return nn.Sequential(*layers)