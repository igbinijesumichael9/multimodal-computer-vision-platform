from app.detection.model_loader import load_yolo_model
from app.utils.image_utils import create_output_directory, save_image


class ObjectDetector:

    def __init__(self, model_name="yolov8n.pt"):

        self.model = load_yolo_model(model_name)

    def detect_objects(self, image_path):

        results = self.model(image_path)

        create_output_directory()

        detections = []

        for result in results:

            annotated_frame = result.plot()

            output_path = "data/outputs/detected_image.jpg"

            save_image(annotated_frame, output_path)

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = self.model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 2)
                })

        return detections, output_path