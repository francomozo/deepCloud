import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,bias = False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, p=0, output_sigmoid=False, filters=64):
        super().__init__()
        self.description = 'Unet_inFrames_' + str(n_channels)+'_outFrames_'+str(n_classes)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2 * filters)
        self.down2 = Down(2 * filters, 4 * filters)
        self.down3 = Down(4 * filters, 8 * filters)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * filters, 16 * filters // factor)
        self.up1 = Up(16 * filters, 8 * filters // factor, bilinear)
        self.up2 = Up(8 * filters, 4 * filters // factor, bilinear)
        self.up3 = Up(4 * filters, 2 * filters // factor, bilinear)
        self.up4 = Up(2 * filters, filters, bilinear)
        self.outc = OutConv(filters, n_classes)
        
        self.dropout2D = nn.Dropout2d(p=p)
        if output_sigmoid:
            self.out_acitvation = nn.Sigmoid()
        else:
            self.out_acitvation = nn.Identity()

    def forward(self, x):
        x1 = self.inc(x)  # convolution (64 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (64 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x2 = self.down1(x1) # maxpool (2x2) => convolution (128 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (128 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x3 = self.down2(x2) # maxpool (2x2) => convolution (256 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (256 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x4 = self.down3(x3) # maxpool (2x2) => convolution (512 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (512 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x5 = self.down4(x4) # maxpool (2x2) => convolution (512 o 1024 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (512 o 1024 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x5 = self.dropout2D(x5)
        x = self.up1(x5, x4) #upsample 
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = self.out_acitvation(out)
            
        return out
        
