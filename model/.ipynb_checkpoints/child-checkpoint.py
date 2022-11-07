from torch import nn
import math

# conv p = math.ceil((k-s)/2)
# tconv op 基本1

class Encoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.lr = nn.LeakyReLU(negative_slope=0.02)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            bias=False,
            padding_mode='zeros',
            )
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            bias=False,
            padding_mode='zeros',
            )
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            bias=False,
            padding_mode='zeros',
            )
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 9),
            stride=(1, 1),
            padding=(1, 4),
            bias=False,
            padding_mode='zeros',
            )
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
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
        
        self.lr = nn.LeakyReLU(negative_slope=0.02)
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            output_padding=0,
            bias=False,
            padding_mode='zeros',
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 9),
            stride=(1, 1),
            padding=(1, 4),
            output_padding=0,
            bias=False,
            padding_mode='zeros',
            )
        self.bn2 = nn.BatchNorm2d(32)
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            output_padding=1,
            bias=False,
            padding_mode='zeros',
            )
        self.bn3 = nn.BatchNorm2d(16)
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            output_padding=1,
            bias=False,
            padding_mode='zeros',
            )
        self.bn4 = nn.BatchNorm2d(8)

        self.tconv5 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=1,
            kernel_size=(3, 9),
            stride=(2, 2),
            padding=(1, 4),
            output_padding=1,
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


class Discriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.lr = nn.LeakyReLU(negative_slope=0.02)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(4*4*32, 256)
        self.linear2 = nn.Linear(256, 100)
        
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.linear1(out)
        out = self.lr(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        
        return out