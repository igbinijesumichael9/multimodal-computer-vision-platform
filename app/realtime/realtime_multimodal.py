import cv2
import pickle
import numpy as np
import face_recognition

from ultralytics import YOLO

# LOAD YOLO MODEL

yolo_model = YOLO("yolov8n.pt")

# LOAD FACE ENCODINGS

with open("data/face_encodings.pkl", "rb") as f:

    data = pickle.load(f)

known_face_encodings = (
    data["encodings"]
)

known_face_names = (
    data["names"]
)

THRESHOLD = 0.50

# START WEBCAM

video_capture = cv2.VideoCapture(0)

print(
    "Starting Real-Time Multimodal AI..."
)

while True:

    # Capture frame
    ret, frame = video_capture.read()

    if not ret:

        break

    # OBJECT DETECTION

    detection_results = yolo_model(frame)

    for result in detection_results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = round(
                float(box.conf[0]),
                2
            )

            class_id = int(box.cls[0])

            class_name = (
                yolo_model.names[class_id]
            )

            # Draw object box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            label = (
                f"{class_name} "
                f"{confidence}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # FACE RECOGNITION

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )

    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    face_locations = (
        face_recognition.face_locations(
            rgb_small_frame
        )
    )

    face_encodings = (
        face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )
    )

    for (
        face_encoding,
        (top, right, bottom, left)
    ) in zip(face_encodings, face_locations):

        face_distances = (
            face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )
        )

        best_match_index = np.argmin(
            face_distances
        )

        best_distance = (
            face_distances[best_match_index]
        )

        name = "Unknown"
        confidence = 0

        if best_distance < THRESHOLD:

            name = known_face_names[
                best_match_index
            ]

            confidence = round(
                float(1 - best_distance),
                2
            )

        # Scale coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw face box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        label = (
            f"{name} "
            f"({confidence})"
        )

        cv2.putText(
            frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # DISPLAY FRAME

    cv2.imshow(
        "Real-Time Multimodal AI",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# Cleanup
video_capture.release()

cv2.destroyAllWindows()