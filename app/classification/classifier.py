import torch
from torchvision import transforms
from PIL import Image

from app.classification.model_loader import (
    load_classification_model
)

from torchvision.models import (
    ResNet50_Weights
)


class ImageClassifier:

    def __init__(self):

        self.weights = ResNet50_Weights.DEFAULT

        self.model = load_classification_model()

        self.categories = self.weights.meta["categories"]

        self.transform = self.weights.transforms()

    def classify_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = self.model(image_tensor)

        probabilities = torch.nn.functional.softmax(
            outputs[0],
            dim=0
        )

        top_prediction = torch.argmax(probabilities).item()

        confidence = probabilities[top_prediction].item()

        predicted_label = self.categories[top_prediction]

        return {
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4)
        }