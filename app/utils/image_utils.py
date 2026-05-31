import os
import cv2


def create_output_directory(path="data/outputs"):
    """
    Create output directory if it doesn't exist.
    """

    os.makedirs(path, exist_ok=True)


def save_image(image, output_path):
    """
    Save image to disk.
    """

    cv2.imwrite(output_path, image)