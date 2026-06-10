import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels)

        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        # FIX: removed stray space — was `super(). __init__()`
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        sk = self.conv(x)   # skip connection (before pooling)
        mp = self.pool(sk)  # downsampled output
        return sk, mp


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up   = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x, sk):
        x = self.up(x)              # upsample
        x = torch.cat([x, sk], dim=1)  # concatenate skip connection
        x = self.conv(x)            # refinement
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels,   features[0])
        self.enc2 = EncoderBlock(features[0],   features[1])
        self.enc3 = EncoderBlock(features[1],   features[2])
        self.enc4 = EncoderBlock(features[2],   features[3])

        self.bn = ConvBlock(features[3], features[3] * 2)

        self.dec1 = DecoderBlock(features[3] * 2, features[3])
        self.dec2 = DecoderBlock(features[3],     features[2])
        self.dec3 = DecoderBlock(features[2],     features[1])
        self.dec4 = DecoderBlock(features[1],     features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        sk1, mp1 = self.enc1(x)    # 64 channels
        sk2, mp2 = self.enc2(mp1)  # 128 channels
        sk3, mp3 = self.enc3(mp2)  # 256 channels
        sk4, mp4 = self.enc4(mp3)  # 512 channels

        # Bottleneck
        b1 = self.bn(mp4)          # 1024 channels

        # Decoder
        up1 = self.dec1(b1,  sk4)  # 512 channels
        up2 = self.dec2(up1, sk3)  # 256 channels
        up3 = self.dec3(up2, sk2)  # 128 channels
        up4 = self.dec4(up3, sk1)  # 64 channels

        return self.final_conv(up4)
