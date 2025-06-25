import torch
import torch.nn as nn
import torch.nn.functional as F

def tanh_normalization(x, alpha=1.0):
    return 0.5 + 0.5 * torch.tanh(alpha * x) # Normalize to [0, 1]


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_le = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # (B, N, C//8)
        proj_key = self.key_conv(x).view(B, -1, H*W)                       # (B, C//8, N)
        energy = torch.bmm(proj_query, proj_key)                             # (B, N, N)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x) + self.value_conv_le(x)             # (B, C, N)
        proj_value = proj_value.view(B, -1, H*W)                     # (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, N)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out



class ResidualBlock(nn.Module):
    """A simple residual block with two convolution layers."""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class Encoder(nn.Module):
    """CNN-based Encoder"""
    def __init__(self, in_channels=4, feature_channels=64):
        super(Encoder, self).__init__()
        self.enc1 = self._block(in_channels, feature_channels)  # (B, 64, 128, 128)
        self.enc2 = self._block(feature_channels, feature_channels * 2)  # (B, 128, 64, 64)
        self.enc3 = self._block(feature_channels * 2, feature_channels * 4)  # (B, 256, 32, 32)
        self.enc4 = self._block(feature_channels * 4, feature_channels * 8)  # (B, 512, 16, 16)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_channels * 8, feature_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_channels * 8),
            nn.ReLU(inplace=True),
            SelfAttention(feature_channels * 8), # Self Attention
            ResidualBlock(feature_channels * 8),  # Residual Block
            SelfAttention(feature_channels * 8), # Self Attention
            ResidualBlock(feature_channels * 8),  # Residual Block
        )  # (B, 512, 16, 16)

    def forward(self, x):
        x1 = self.enc1(x)  # Save for skip connections
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        bottleneck = self.bottleneck(x4)
        return bottleneck, [x1, x2, x3, x4]

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)  # Residual Block
        )

class Decoder(nn.Module):
    """CNN-based Decoder"""
    def __init__(self, out_channels=3, feature_channels=64):
        super(Decoder, self).__init__()
        self.up4 = self._up_block(feature_channels * 8, feature_channels * 4)  # 16x16 -> 32x32
        self.up3 = self._up_block(feature_channels * 4, feature_channels * 2)  # 32x32 -> 64x64
        self.up2 = self._up_block(feature_channels * 2, feature_channels * 1)  # 64x64 -> 128x128
        self.up1 = self._up_block(feature_channels * 1, feature_channels)  # 128x128 -> 256x256
        self.final = nn.Sequential(
                    nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                    )
        

    def forward(self, x, encoder_features):
        x = self.up4(x + encoder_features[3])  # Skip connection
        x = self.up3(x + encoder_features[2])
        x = self.up2(x + encoder_features[1])
        x = self.up1(x + encoder_features[0])
        # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # Upsample
        x = self.final(x)
        return tanh_normalization(x)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)  # Residual Block
        )

class AutoEncoder(nn.Module):
    """Parallel CNN-based Autoencoder (Separate Sim & Real branches)"""
    def __init__(self, in_channels=4, out_channels=4, feature_channels=64):
        super(AutoEncoder, self).__init__()
        self.sim_encoder = Encoder(in_channels, feature_channels)
        self.real_encoder = Encoder(in_channels, feature_channels)
        self.sim_decoder = Decoder(out_channels, feature_channels)
        self.real_decoder = Decoder(out_channels, feature_channels)

    def forward(self, x, real=True):
        if real:
            x_clone = x.clone()  # Clone input
            x_clone[:, :3, :, :] = 0  # Set depth to 0
            bottleneck, encoder_features = self.real_encoder(x_clone)
            return self.real_decoder(bottleneck, encoder_features)
        else:
            x_clone = x.clone()  # Clone input
            x_clone[:, 3, :, :] = 0  # Set depth to 0
            bottleneck, encoder_features = self.sim_encoder(x_clone)
            return self.sim_decoder(bottleneck, encoder_features)



class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64):
        super(PatchDiscriminator, self).__init__()
        self.model_real = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels * 2, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, real=True):
        

        return self.model_real(x)
