from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
import math

# conv p = (k-s)//2
# tconv op 基本1

class AttributeEncoder(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        
        params = AttributeDict()
        params.update(layer1=AttributeDict())
        params.layer1.update(channels=1,
                             kernel_size=(1, 9),
                             stride=(1, 1),
                             padding=(0, 4)
                            )
        
        self.save_hyperparameters(params)
        
        layer2_inchannels = 8
        layer2_kernel_size = (1, 9)
        layer2_stride = (2, 2)
        layer2_padding =  (math.ceil((layer2_kernel_size[0]-layer2_stride[0])/2), math.ceil((layer2_kernel_size[1]-layer2_stride[1])/2))
        
        layer3_inchannels = 16
        layer3_kernel_size = (1, 9)
        layer3_stride = (2, 2)
        layer3_padding =  (math.ceil((layer3_kernel_size[0]-layer3_stride[0])/2), math.ceil((layer3_kernel_size[1]-layer3_stride[1])/2))
        
        layer4_inchannels = 32
        layer4_kernel_size = (1, 9)
        layer4_stride = (2, 2)
        layer4_padding = (math.ceil((layer4_kernel_size[0]-layer4_stride[0])/2), math.ceil((layer4_kernel_size[1]-layer4_stride[1])/2))
        
        layer5_inchannels = 32
        layer5_kernel_size = (9, 1)
        layer5_stride = (1, 1)
        layer5_padding = (math.ceil((layer5_kernel_size[0]-layer5_stride[0])/2), math.ceil((layer5_kernel_size[1]-layer5_stride[1])/2))
        
        out_channels = 64
        
        
        self.lr = nn.GLU(dim=1)
        self.conv1 = nn.Conv2d(
            in_channels=params.layer1.channels,
            out_channels=layer2_inchannels*2,
            kernel_size=params.layer1.kernel_size,
            stride=params.layer1.stride,
            padding=params.layer1.padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn1 = nn.BatchNorm2d(layer2_inchannels*2)
        self.conv2 = nn.Conv2d(
            in_channels=layer2_inchannels,
            out_channels=layer3_inchannels*2,
            kernel_size=layer2_kernel_size,
            stride=layer2_stride,
            padding=layer2_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn2 = nn.BatchNorm2d(layer3_inchannels*2)
        self.conv3 = nn.Conv2d(
            in_channels=layer3_inchannels,
            out_channels=layer4_inchannels*2,
            kernel_size=layer3_kernel_size,
            stride=layer3_stride,
            padding=layer3_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn3 = nn.BatchNorm2d(layer4_inchannels*2)
        self.conv4 = nn.Conv2d(
            in_channels=layer4_inchannels,
            out_channels=layer5_inchannels*2,
            kernel_size=layer4_kernel_size,
            stride=layer4_stride,
            padding=layer4_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn4 = nn.BatchNorm2d(layer5_inchannels*2)
        self.conv5 = nn.Conv2d(
            in_channels=layer5_inchannels,
            out_channels=out_channels,
            kernel_size=layer5_kernel_size,
            stride=layer5_stride,
            padding=layer5_padding,
            bias=False,
            padding_mode='zeros',
            )

        


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lr(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lr(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lr(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.lr(out)
        out = self.conv5(out)

        return out
    

class ContentEncoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        
        layer1_inchannels = 1
        layer1_kernel_size = (3, 9)
        layer1_stride = (1, 1)
        layer1_padding =  (math.ceil((layer1_kernel_size[0]-layer1_stride[0])/2), math.ceil((layer1_kernel_size[1]-layer1_stride[1])/2))
        
        layer2_inchannels = 8
        layer2_kernel_size = (3, 9)
        layer2_stride = (2, 2)
        layer2_padding =  (math.ceil((layer2_kernel_size[0]-layer2_stride[0])/2), math.ceil((layer2_kernel_size[1]-layer2_stride[1])/2))
        
        layer3_inchannels = 16
        layer3_kernel_size = (3, 9)
        layer3_stride = (2, 2)
        layer3_padding =  (math.ceil((layer3_kernel_size[0]-layer3_stride[0])/2), math.ceil((layer3_kernel_size[1]-layer3_stride[1])/2))
        
        layer4_inchannels = 32
        layer4_kernel_size = (3, 9)
        layer4_stride = (2, 2)
        layer4_padding = (math.ceil((layer4_kernel_size[0]-layer4_stride[0])/2), math.ceil((layer4_kernel_size[1]-layer4_stride[1])/2))
        
        layer5_inchannels = 32
        layer5_kernel_size = (9, 3)
        layer5_stride = (1, 1)
        layer5_padding = (math.ceil((layer5_kernel_size[0]-layer5_stride[0])/2), math.ceil((layer5_kernel_size[1]-layer5_stride[1])/2))
        
        out_channels = 64
        
        
        self.lr = nn.GLU(dim=1)
        self.conv1 = nn.Conv2d(
            in_channels=layer1_inchannels,
            out_channels=layer2_inchannels*2,
            kernel_size=layer1_kernel_size,
            stride=layer1_stride,
            padding=layer1_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn1 = nn.BatchNorm2d(layer2_inchannels*2)
        self.conv2 = nn.Conv2d(
            in_channels=layer2_inchannels,
            out_channels=layer3_inchannels*2,
            kernel_size=layer2_kernel_size,
            stride=layer2_stride,
            padding=layer2_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn2 = nn.BatchNorm2d(layer3_inchannels*2)
        self.conv3 = nn.Conv2d(
            in_channels=layer3_inchannels,
            out_channels=layer4_inchannels*2,
            kernel_size=layer3_kernel_size,
            stride=layer3_stride,
            padding=layer3_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn3 = nn.BatchNorm2d(layer4_inchannels*2)
        self.conv4 = nn.Conv2d(
            in_channels=layer4_inchannels,
            out_channels=layer5_inchannels*2,
            kernel_size=layer4_kernel_size,
            stride=layer4_stride,
            padding=layer4_padding,
            bias=False,
            padding_mode='zeros',
            )
        self.bn4 = nn.BatchNorm2d(layer5_inchannels*2)
        self.conv5 = nn.Conv2d(
            in_channels=layer5_inchannels,
            out_channels=out_channels,
            kernel_size=layer5_kernel_size,
            stride=layer5_stride,
            padding=layer5_padding,
            bias=False,
            padding_mode='zeros',
            )

        


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lr(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lr(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lr(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.lr(out)
        out = self.conv5(out)

        return out
    

class Decoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        
        self.lr = nn.GLU(dim=1)
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(9, 3),
            stride=(1, 1),
            padding=(4, 1), 
            output_padding=0,
            bias=False,
            padding_mode='zeros',
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            output_padding=1,
            bias=False,
            padding_mode='zeros',
            )
        self.bn2 = nn.BatchNorm2d(64)
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            output_padding=1,
            bias=False,
            padding_mode='zeros',
            )
        self.bn3 = nn.BatchNorm2d(32)
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            output_padding=1,
            bias=False,
            padding_mode='zeros',
            )
        self.bn4 = nn.BatchNorm2d(16)

        self.tconv5 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=1,
            kernel_size=(3, 9),
            stride=(1, 1),
            padding=(1, 4),
            output_padding=0,
            bias=False,
            padding_mode='zeros',
            )


    def forward(self, x):
        out = self.tconv1(x)
        out = self.bn1(out)
        out = self.lr(out)
        out = self.tconv2(out)
        out = self.bn2(out)
        out = self.lr(out)
        out = self.tconv3(out)
        out = self.bn3(out)
        out = self.lr(out)
        out = self.tconv4(out)
        out = self.bn4(out)
        out = self.lr(out)
        out = self.tconv5(out)
        
        return out


class AttributeDiscriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.lr = nn.LeakyReLU(negative_slope=0.02)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(4*4*32, 256)
        self.linear2 = nn.Linear(256, 51)
        
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.linear1(out)
        out = self.lr(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        
        return out
    
class ContentDiscriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.lr = nn.LeakyReLU(negative_slope=0.02)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(4*4*32, 256)
        self.linear2 = nn.Linear(256, 51)
        
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.linear1(out)
        out = self.lr(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        
        return out