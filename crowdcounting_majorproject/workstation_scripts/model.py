import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNetV3Large_Weights

class MobileNetV3MultiScale(nn.Module):
    def __init__(self):
        super(MobileNetV3MultiScale, self).__init__()
        self.mobilenet_features = mobilenet_v3_large(weights=MobileNetV3Large_Weights.DEFAULT).features
        self.feature_extraction_points = [6, 12, 16]

    def forward(self, x):
        features = []
        for i, module in enumerate(self.mobilenet_features):
            x = module(x)
            if i in self.feature_extraction_points:
                features.append(x)
        return features

class AdaptiveFusionModule(nn.Module):
    def __init__(self, in_channels_list=[40, 112, 960], out_channels=128): # 960 fix applied!
        super(AdaptiveFusionModule, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) 
            for in_channels in in_channels_list
        ])
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(in_channels_list[-1], len(in_channels_list)), 
            nn.Softmax(dim=1) 
        )

    def forward(self, features):
        weights = self.weight_predictor(features[-1])
        fused_features = []
        target_size = features[0].shape[2:] 

        for i, feat in enumerate(features):
            processed_feat = self.convs[i](feat)
            if processed_feat.shape[2:] != target_size:
                processed_feat = F.interpolate(processed_feat, size=target_size, mode='bilinear', align_corners=False)
            weight_i = weights[:, i].view(-1, 1, 1, 1)
            weighted_feat = processed_feat * weight_i
            fused_features.append(weighted_feat)
        return sum(fused_features)

class MyMAN(nn.Module):
    def __init__(self, in_channels):
        super(MyMAN, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_map = self.channel_attention(x)
        x = x * ca_map
        sa_map = self.spatial_attention(x)
        x = x * sa_map
        return x

class EdgeCrowdCounter(nn.Module):
    def __init__(self, num_output_channels=1):
        super(EdgeCrowdCounter, self).__init__()
        self.backbone = MobileNetV3MultiScale()
        in_channels_list = [40, 112, 960] 
        fused_channels = 128 
        self.adaptive_fusion = AdaptiveFusionModule(in_channels_list, fused_channels)
        self.man = MyMAN(fused_channels)
        self.density_map_head = nn.Sequential(
            nn.Conv2d(fused_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_output_channels, kernel_size=1),
            nn.ReLU(inplace=True) 
        )

    def forward(self, x):
        features = self.backbone(x)
        fused_output = self.adaptive_fusion(features)
        man_output = self.man(fused_output)
        density_map = self.density_map_head(man_output)
        return density_map