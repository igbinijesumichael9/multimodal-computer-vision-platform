import cv2
import pickle
import numpy as np
import face_recognition

ENCODINGS_PATH = "data/face_encodings.pkl"


# LOAD ENCODINGS

with open(ENCODINGS_PATH, "rb") as f:

    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Recognition threshold
THRESHOLD = 0.50

# START WEBCAM

video_capture = cv2.VideoCapture(0)

print("Starting webcam face recognition...")

while True:

    # Capture frame
    ret, frame = video_capture.read()

    if not ret:

        break

    # Resize frame for faster inference
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )

    # Convert BGR → RGB
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect faces
    face_locations = (
        face_recognition.face_locations(
            rgb_small_frame
        )
    )

    # Extract embeddings
    face_encodings = (
        face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )
    )

    # Process each face
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

        # Default values
        name = "Unknown"
        confidence = 0

        # Match found
        if best_distance < THRESHOLD:

            name = known_face_names[
                best_match_index
            ]

            confidence = round(
                float(1 - best_distance),
                2
            )

        # Scale coordinates back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Label
        label = (
            f"{name} ({confidence})"
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

    # Show webcam feed
    cv2.imshow(
        "Real-Time Face Recognition",
        frame
    )

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# Cleanup
video_capture.release()

cv2.destroyAllWindows()