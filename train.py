import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import MAGSwin
from dataset import MRIDataset
from utils import set_seed


def train():

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    train_dataset = MRIDataset(
        cfg["paths"]["train_csv"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    model = MAGSwin(
        num_classes=cfg["dataset"]["num_classes"]
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cfg["training"]["learning_rate"]
    )

    for epoch in range(cfg["training"]["epochs"]):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1} Loss:"
            f"{running_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "checkpoints/magswin_best.pth"
    )


if __name__ == "__main__":
    train()
