from app.detection.detector import ObjectDetector


if __name__ == "__main__":

    image_path = "data/samples/street_scenes.jpg"

    detector = ObjectDetector()

    detections, output_path = detector.detect_objects(image_path)

    print("\nDetected Objects:\n")

    for detection in detections:

        print(
            f"{detection['class']} - "
            f"Confidence: {detection['confidence']}"
        )

    print(f"\nOutput saved to: {output_path}")