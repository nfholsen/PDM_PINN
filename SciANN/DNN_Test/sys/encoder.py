import torch
import torch.nn as nn
from collections import OrderedDict

class AutoEncoder(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super(AutoEncoder, self).__init__()

        ###################
        # # # Encoder # # #
        ###################

        features = init_features
        self.encoder1 = AutoEncoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = AutoEncoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = AutoEncoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = AutoEncoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = AutoEncoder._block(features * 8, features * 16, name="bottleneck")

        ###################
        # # # Decoder # # #
        ###################

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2, output_padding = 1 # Add output_padding to match that 37 is not divisible by 2 in the encoder
        )
        self.decoder4 = AutoEncoder._block(features * 8, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2, output_padding = 1 # Add output_padding to match that 75 is not divisible by 2 in the encoder
        )
        self.decoder3 = AutoEncoder._block(features * 4, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = AutoEncoder._block(features * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = AutoEncoder._block(features, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        ###################
        # # # Encoder # # #
        ###################

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        ###################
        # # # Decoder # # #
        ###################

        dec4 = self.upconv4(bottleneck)
        dec3 = self.upconv3(self.decoder4(dec4))
        dec2 = self.upconv2(self.decoder3(dec3))
        dec1 = self.upconv1(self.decoder2(dec2))
        dec1 = self.decoder1(dec1)

        return self.conv(dec1) # No activation function because we want to return the real value, not a prediction between 0 and 1 if the sigmoid was used

    @staticmethod # Static method to define each block in the AutoEncoder
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )