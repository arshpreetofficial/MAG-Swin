import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=256):
        super(AttentionFusion, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, spectral, spatial, anatomical):
        features = torch.stack(
            [spectral, spatial, anatomical],
            dim=1
        )

        attention_scores = self.attention(features)
        attention_weights = torch.softmax(attention_scores, dim=1)

        fused = torch.sum(attention_weights * features, dim=1)
        return fused
