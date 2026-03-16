import torch
import torch.nn as nn
import torch.nn.functional as F

class RepBlock(nn.Module):
    """
    Reparameterization Block inspired by RepVGG and RepSegNet (2025/2026).
    During training, it uses a multi-branch architecture (3x3 conv, 1x1 conv, identity).
    In inference, this could theoretically be fused into a single 3x3 conv for speed.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels else None
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out3 = self.bn3(self.conv3x3(x))
        out1 = self.bn1(self.conv1x1(x))
        id_out = self.identity(x) if self.identity is not None else 0
        return self.relu(out3 + out1 + id_out)

class RepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.rep1 = RepBlock(in_channels, out_channels)
        self.rep2 = RepBlock(out_channels, out_channels)

    def forward(self, x):
        return self.rep2(self.rep1(x))

class ECA_Attention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Often used in modern lightweight medical segmentation models (2025/2026).
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class RepSegNet(nn.Module):
    """
    RepSegNet: A modern (2026) lightweight reparameterized U-Net for medical image segmentation.
    Combines RepBlocks for feature extraction and ECA for channel attention.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        in_c = in_channels
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    RepConvBlock(in_c, feature),
                    ECA_Attention(feature)
                )
            )
            in_c = feature
            
        # Bottleneck
        self.bottleneck = RepConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                nn.Sequential(
                    RepConvBlock(feature * 2, feature),
                    ECA_Attention(feature)
                )
            )
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)
            
        return torch.sigmoid(self.final_conv(x))

def get_repsegnet(in_channels=3, out_channels=1):
    return RepSegNet(in_channels, out_channels)

if __name__ == "__main__":
    net = get_repsegnet()
    x = torch.randn(2, 3, 128, 128)
    y = net(x)
    print("RepSegNet output shape:", y.shape) # Should be [2, 1, 128, 128]
    print("Parameters:", sum(p.numel() for p in net.parameters()))
