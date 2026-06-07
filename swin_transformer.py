import torch.nn as nn
import timm


class SwinSpatialEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super(SwinSpatialEncoder, self).__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            in_chans=1
        )

        self.fc = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.fc(feat)
        return feat
