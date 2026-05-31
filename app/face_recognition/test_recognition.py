from app.face_recognition.recognizer import (
    FaceRecognizer
)

if __name__ == "__main__":

    image_path = (
        "data/faces/test_faces/Eloho_test.jpg"
    )

    recognizer = FaceRecognizer()

    result = recognizer.recognize_faces(
        image_path
    )

    print("\nRecognition Result:\n")

    print(result)