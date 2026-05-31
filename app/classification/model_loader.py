from torchvision import models


def load_classification_model():
    """
    Load pretrained ResNet50 model.
    """

    weights = models.ResNet50_Weights.IMAGENET1K_V2

    model = models.resnet50(
        weights=weights
    )

    model.eval()

    return model