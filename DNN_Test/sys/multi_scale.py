import torch
import torch.nn as nn

class ConvBlockSmall(nn.Module): 
    def __init__(self,in_channels,in_layers=32): 
        super(ConvBlockSmall, self).__init__()

        self.convblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_layers, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers, out_channels=in_layers*2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers*2, out_channels=in_layers, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.convblock(x)
        return x

class ConvBlockBig(nn.Module): 
    def __init__(self,in_channels,in_layers=32): 
        super(ConvBlockBig, self).__init__()

        self.convblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_layers, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers, out_channels=in_layers*2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers*2, out_channels=in_layers*4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers*4, out_channels=in_layers*2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers*2, out_channels=in_layers, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=in_layers, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.convblock(x)
        return x

class MultiScale(nn.Module):
    def __init__(self,in_channels,in_layers=32): 
        super(MultiScale, self).__init__()

        self.convblock_1 = ConvBlockSmall(in_channels=in_channels,in_layers=in_layers)
        self.convblock_2 = ConvBlockBig(in_channels=in_channels+1,in_layers=in_layers)
        self.convblock_3 = ConvBlockBig(in_channels=in_channels+1,in_layers=in_layers)

        self.upsample = nn.Upsample(scale_factor=(2,2))

    def forward(self,x_4,x_2,x_1):

        x = self.convblock_1(x_4)
        x = self.upsample(x)

        x = self.convblock_2(torch.cat((x,x_2),dim=1))
        x = self.upsample(x)

        x = self.convblock_2(torch.cat((x,x_1),dim=1))

        return x