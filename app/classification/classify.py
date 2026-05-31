from app.classification.classifier import (
    ImageClassifier
)


if __name__ == "__main__":

    image_path = "data/samples/street_scenes.jpg"

    classifier = ImageClassifier()

    result = classifier.classify_image(image_path)

    print("\nClassification Result:\n")

    print(result)