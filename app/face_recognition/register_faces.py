import os
import pickle

from app.face_recognition.face_encoder import (
    encode_faces
)

KNOWN_FACES_DIR = "data/faces"

ENCODINGS_PATH = "data/face_encodings.pkl"


known_face_encodings = []
known_face_names = []


for person_name in os.listdir(KNOWN_FACES_DIR):

    person_dir = os.path.join(
        KNOWN_FACES_DIR,
        person_name
    )

    if not os.path.isdir(person_dir):

        continue

    for filename in os.listdir(person_dir):

        image_path = os.path.join(
            person_dir,
            filename
        )

        print(f"Encoding {image_path}")

        encodings = encode_faces(image_path)

        if len(encodings) > 0:
            for encoding in encodings:

                known_face_encodings.append(
                    encoding
                )

                known_face_names.append(
                    person_name
                )

        else:

            print(
                f"No face found in {image_path}"
            )

# Save encodings
data = {
    "encodings": known_face_encodings,
    "names": known_face_names
}

with open(ENCODINGS_PATH, "wb") as f:

    pickle.dump(data, f)

print("\nFace encodings saved successfully.")