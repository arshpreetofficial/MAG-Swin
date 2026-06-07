import cv2
import torch

from model import MAGSwin


classes = [
    "CN",
    "MCI",
    "AD"
]

model = MAGSwin(num_classes=3)

model.load_state_dict(
    torch.load(
        "checkpoints/magswin_best.pth"
    )
)

model.eval()


def predict(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image / 255.0

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        pred = output.argmax(1)

    return classes[pred.item()]
