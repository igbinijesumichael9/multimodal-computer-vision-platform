import face_recognition


def encode_faces(image_path):

    # Load image
    image = face_recognition.load_image_file(
        image_path
    )

    # Detect face locations
    face_locations = (
        face_recognition.face_locations(image)
    )

    # Extract encodings for ALL faces
    encodings = (
        face_recognition.face_encodings(
            image,
            face_locations
        )
    )

    return encodings