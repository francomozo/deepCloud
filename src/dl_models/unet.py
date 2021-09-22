import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels # In Down mid_channels = out_channels 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_w_options(nn.Module):
    """(convolution => [BN] => Activation) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False, activation='relu',
                 normalization='batch', img_shape=(256, 256)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels # In Down mid_channels = out_channels
            
        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias)]
        
        if normalization == 'batch':          
            layers.append(nn.BatchNorm2d(mid_channels))
        if normalization == 'layer':
            layers.append(nn.LayerNorm(normalized_shape=(mid_channels, img_shape[0], img_shape[1])))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias))
            if normalization == 'batch':          
                layers.append(nn.BatchNorm2d(out_channels))
            if normalization == 'layer':
                layers.append(nn.LayerNorm(normalized_shape=(out_channels, img_shape[0], img_shape[1])))
            layers.append(nn.ReLU(inplace=True))
            
        if activation in ['leaky_relu', 'leaky', 'LeakyReLU', 'leakyrelu']:
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias))
            if normalization == 'batch':          
                layers.append(nn.BatchNorm2d(out_channels))
            if normalization == 'layer':
                layers.append(nn.LayerNorm(normalized_shape=(out_channels, img_shape[0], img_shape[1])))
            layers.append(nn.LeakyReLU(inplace=True))
            
        if activation in ['prelu', 'PRelu', 'PRELU', 'PReLU']:
            layers.append(nn.PReLU())
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias))
            if normalization == 'batch':          
                layers.append(nn.BatchNorm2d(out_channels))
            if normalization == 'layer':
                layers.append(nn.LayerNorm(normalized_shape=(out_channels, img_shape[0], img_shape[1])))
            layers.append(nn.PReLU())
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down_w_options(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False, activation='relu',
                 normalization='batch', img_shape=(0,0)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_w_options(in_channels, out_channels, bias=bias, activation=activation, normalization=normalization,
                                 img_shape=img_shape)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, bias=bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, bias=bias)


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


class Up_w_options(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bias=False, activation='relu',
                 normalization='batch', img_shape=(0,0)):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_w_options(in_channels, out_channels, in_channels // 2, bias=bias, activation=activation,
                                             normalization=normalization, img_shape=img_shape)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_w_options(in_channels, out_channels, bias=bias, activation=activation,
                                             normalization=normalization, img_shape=img_shape)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, p=0, output_activation='sigmoid', filters=64, bias=False):
        super().__init__()
        self.description = 'Unet_inFrames_' + str(n_channels)+'_outFrames_'+str(n_classes)+'_out_activation'+str(output_activation)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2 * filters, bias=bias)
        self.down2 = Down(2 * filters, 4 * filters, bias=bias)
        self.down3 = Down(4 * filters, 8 * filters, bias=bias)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * filters, 16 * filters // factor, bias=bias)
        self.up1 = Up(16 * filters, 8 * filters // factor, bilinear, bias=bias)
        self.up2 = Up(8 * filters, 4 * filters // factor, bilinear, bias=bias)
        self.up3 = Up(4 * filters, 2 * filters // factor, bilinear, bias=bias)
        self.up4 = Up(2 * filters, filters, bilinear, bias=bias)
        self.outc = OutConv(filters, n_classes)
        
        self.dropout2D = nn.Dropout2d(p=p)
        if output_activation:
            if output_activation in ['sigmoid', 'Sigmoid', 'sigmoide', 'Sigmoide', 'sig']:
                self.out_activation = nn.Sigmoid()
            if output_activation in ['relu', 'ReLu', 'Relu']:
                self.out_activation = nn.Hardtanh(min_val=0, max_val=1.0)  #works as relu clip between [0,1]
            if output_activation in ['tanh']:
                self.out_activation = nn.Tanh()
        else:
            self.out_activation = nn.Identity()

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
        out = self.out_activation(out)
        return out
        
class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, p=0, output_activation='sigmoid', filters=64, bias=False):
        super().__init__()
        self.description = 'Unet2_inFrames' + str(n_channels)+'_outFrames'+str(n_classes)+'_out_activation'+str(output_activation)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, filters, bias=bias)
        self.down1 = Down(filters, 2 * filters, bias=bias)
        self.down2 = Down(2 * filters, 4 * filters, bias=bias)
        self.down3 = Down(4 * filters, 8 * filters, bias=bias)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * filters, 16 * filters // factor, bias=bias)
        self.up1 = Up(16 * filters, 8 * filters // factor, bilinear, bias=bias)
        self.up2 = Up(8 * filters, 4 * filters // factor, bilinear, bias=bias)
        self.up3 = Up(4 * filters, 2 * filters // factor, bilinear, bias=bias)
        self.up4 = Up(2 * filters, filters, bilinear, bias=bias)
        self.outc = OutConv(filters+n_channels, n_classes)
        
        self.dropout2D = nn.Dropout2d(p=p)
        if output_activation:
            if output_activation in ['sigmoid', 'Sigmoid', 'sigmoide', 'Sigmoide', 'sig']:
                self.out_activation = nn.Sigmoid()
            if output_activation in ['relu', 'ReLu', 'Relu']:
                self.out_activation = nn.Hardtanh(min_val=0, max_val=1.0)  #works as relu clip between [0,1]
            if output_activation in ['tanh']:
                self.out_activation = nn.Tanh()
        else:
            self.out_activation = nn.Identity()

    def forward(self, x_in):
        x1 = self.inc(x_in)  # convolution (64 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (64 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x2 = self.down1(x1) # maxpool (2x2) => convolution (128 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (128 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x2 = self.dropout2D(x2)
        x3 = self.down2(x2) # maxpool (2x2) => convolution (256 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (256 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x3 = self.dropout2D(x3)
        x4 = self.down3(x3) # maxpool (2x2) => convolution (512 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (512 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x4 = self.dropout2D(x4)
        x5 = self.down4(x4) # maxpool (2x2) => convolution (512 o 1024 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (512 o 1024 filters 3x3, pad=1 )=> [BN] => ReLU) 
        x5 = self.dropout2D(x5)
        x6 = self.up1(x5, x4) #upsample 
        x6 = self.dropout2D(x6)
        x7 = self.up2(x6, x3)
        x7 = self.dropout2D(x7)
        x8 = self.up3(x7, x2)
        x8 = self.dropout2D(x8)
        x9 = self.up4(x8, x1)
        x_out = torch.cat([x9, x_in], dim=1)
        x_out = self.outc(x_out)
        
        x_out = self.out_activation(x_out)
        return x_out

class UNet_w_options(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, p=0, output_activation='sigmoid', filters=64, bias=False,
                 internal_activations='relu', normalization='batch', batch_size=1, img_shape=(256,256)):
        super().__init__()
        self.description = 'Unet_inFrames_' + str(n_channels)+'_outFrames_'+str(n_classes)+'_out_activation'+str(output_activation)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.img_shape = img_shape

        self.inc = DoubleConv_w_options(n_channels, filters, activation=internal_activations,
                                        normalization=normalization, img_shape=self.img_shape)
        
        self.down1 = Down_w_options(filters, 2 * filters, bias=bias, activation=internal_activations,
                                    normalization=normalization, img_shape=(self.img_shape[0]//2, self.img_shape[1]//2))
        
        self.down2 = Down_w_options(2 * filters, 4 * filters, bias=bias, activation=internal_activations,
                                    normalization=normalization, img_shape=(self.img_shape[0]//4, self.img_shape[1]//4))
        
        self.down3 = Down_w_options(4 * filters, 8 * filters, bias=bias, activation=internal_activations,
                                    normalization=normalization, img_shape=(self.img_shape[0]//8, self.img_shape[1]//8))
        
        factor = 2 if bilinear else 1
        self.down4 = Down_w_options(8 * filters, 16 * filters // factor, bias=bias, activation=internal_activations,
                                    normalization=normalization, img_shape=(self.img_shape[0]//16, self.img_shape[1]//16))
        
        self.up1 = Up_w_options(16 * filters, 8 * filters // factor, bilinear, bias=bias, activation=internal_activations,
                                normalization=normalization, img_shape=(self.img_shape[0]//8, self.img_shape[1]//8))
        
        self.up2 = Up_w_options(8 * filters, 4 * filters // factor, bilinear, bias=bias, activation=internal_activations,
                                normalization=normalization, img_shape=(self.img_shape[0]//4, self.img_shape[1]//4))
        
        self.up3 = Up_w_options(4 * filters, 2 * filters // factor, bilinear, bias=bias, activation=internal_activations,
                                normalization=normalization, img_shape=(self.img_shape[0]//2, self.img_shape[1]//2))
        
        self.up4 = Up_w_options(2 * filters, filters, bilinear, bias=bias, activation=internal_activations,
                                normalization=normalization, img_shape=self.img_shape)
        
        self.outc = OutConv(filters, n_classes)
        
        self.dropout2D = nn.Dropout2d(p=p)
        if output_activation:
            if output_activation in ['sigmoid', 'Sigmoid', 'sigmoide', 'Sigmoide', 'sig']:
                self.out_activation = nn.Sigmoid()
            if output_activation in ['relu', 'ReLu', 'Relu']:
                self.out_activation = nn.Hardtanh(min_val=0, max_val=1.0)  #works as relu clip between [0,1]
            if output_activation in ['tanh']:
                self.out_activation = nn.Tanh()
        else:
            self.out_activation = nn.Identity()

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
        out = self.out_activation(out)
        return out
