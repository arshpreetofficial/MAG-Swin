import torch
import torch.nn as nn

from spectral_encoder import SpectralEncoder
from swin_transformer import SwinSpatialEncoder
from anatomical_branch import AnatomicalBranch
from fusion_module import AttentionFusion


class MAGSwin(nn.Module):
    def __init__(self, num_classes=3, feature_dim=256, dropout=0.3):
        super(MAGSwin, self).__init__()

        self.spectral_encoder = SpectralEncoder(out_dim=feature_dim)
        self.spatial_encoder = SwinSpatialEncoder(out_dim=feature_dim)
        self.anatomical_encoder = AnatomicalBranch(out_dim=feature_dim)

        self.fusion = AttentionFusion(feature_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        spectral_feat = self.spectral_encoder(x)
        spatial_feat = self.spatial_encoder(x)
        anatomical_feat = self.anatomical_encoder(x)

        fused_feat = self.fusion(
            spectral_feat,
            spatial_feat,
            anatomical_feat
        )

        output = self.classifier(fused_feat)
        return output
