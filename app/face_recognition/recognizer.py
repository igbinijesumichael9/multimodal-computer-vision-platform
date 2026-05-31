import cv2
import pickle
import numpy as np
import face_recognition

ENCODINGS_PATH = "data/face_encodings.pkl"


class FaceRecognizer:

    def __init__(self):

        with open(ENCODINGS_PATH, "rb") as f:

            data = pickle.load(f)

        self.known_face_encodings = (
            data["encodings"]
        )

        self.known_face_names = (
            data["names"]
        )

        self.threshold = 0.50

    def recognize_faces(self, image_path):

        # Load image
        image = face_recognition.load_image_file(
            image_path
        )

        # Convert for OpenCV
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # Detect face locations
        face_locations = (
            face_recognition.face_locations(image)
        )

        # Extract encodings
        face_encodings = (
            face_recognition.face_encodings(
                image,
                face_locations
            )
        )

        results = []

        # Process each face
        for (
            face_encoding,
            (top, right, bottom, left)
        ) in zip(face_encodings, face_locations):

            face_distances = (
                face_recognition.face_distance(
                    self.known_face_encodings,
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
            person_name = "Unknown"
            confidence = 0

            # Match found
            if best_distance < self.threshold:

                person_name = (
                    self.known_face_names[
                        best_match_index
                    ]
                )

                confidence = round(
                    float(1 - best_distance),
                    4
                )

            # Save result
            results.append({
                "person": person_name,
                "confidence": confidence
            })

            # DRAW FACE BOX
            cv2.rectangle(
                image_bgr,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            # LABEL TEXT
            label = (
                f"{person_name} "
                f"({confidence})"
            )

            cv2.putText(
                image_bgr,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Save annotated image
        output_path = (
            "data/outputs/recognized_faces.jpg"
        )

        cv2.imwrite(
            output_path,
            image_bgr
        )

        return results, output_path