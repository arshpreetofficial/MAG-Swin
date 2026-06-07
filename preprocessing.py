import cv2
import numpy as np


def clahe_enhancement(image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)
    return enhanced


def normalize(image):
    image = image.astype(np.float32)
    image = (image - image.min()) / (
        image.max() - image.min() + 1e-8
    )
    return image
