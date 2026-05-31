from torchvision import models

print("Downloading ResNet50 model...")

weights = models.ResNet50_Weights.IMAGENET1K_V2

model = models.resnet50(
    weights=weights
)

print("Model downloaded successfully.")