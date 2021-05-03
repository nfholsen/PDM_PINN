import torch
import torch.nn as nn

class ConvBlock(nn.Module): 
    def __init__(self,in_channels): 
        super(ConvBlock, self).__init__()

        self.convblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.convblock(x)
        return x

class MultiScale(nn.Module):
    def __init__(self,in_channels): 
        super(MultiScale, self).__init__()

        self.convblock_1 = ConvBlock(in_channels=in_channels)
        self.convblock_2 = ConvBlock(in_channels=in_channels+1)
        self.convblock_3 = ConvBlock(in_channels=in_channels+1)

        self.upsample = nn.Upsample(scale_factor=(2,2))

    def forward(self,x_4,x_2,x_1):

        x = self.convblock_1(x_4)
        x = self.upsample(x)

        x = self.convblock_2(torch.cat((x,x_2),dim=1))
        x = self.upsample(x)

        x = self.convblock_2(torch.cat((x,x_1),dim=1))

        return x