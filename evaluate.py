import yaml
import torch

from model import MAGSwin
from dataset import MRIDataset
from metrics import compute_metrics

from torch.utils.data import DataLoader


def evaluate():

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset = MRIDataset(
        cfg["paths"]["test_csv"]
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False
    )

    model = MAGSwin(
        num_classes=cfg["dataset"]["num_classes"]
    )

    model.load_state_dict(
        torch.load(
            "checkpoints/magswin_best.pth"
        )
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            preds = outputs.argmax(1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    metrics = compute_metrics(
        y_true,
        y_pred
    )

    print(metrics)


if __name__ == "__main__":
    evaluate()
