from ultralytics import YOLO


def load_yolo_model(model_name="yolov8n.pt"):
    """
    Load YOLO model.
    """

    model = YOLO(model_name)

    return model