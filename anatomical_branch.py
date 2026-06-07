import torch.nn as nn


class AnatomicalBranch(nn.Module):
    def __init__(self, out_dim=256):
        super(AnatomicalBranch, self).__init__()

        self.branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        feat = self.branch(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return feat
