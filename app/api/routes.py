from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.detection.detector import ObjectDetector
from app.classification.classifier import ImageClassifier
from app.face_recognition.recognizer import (
    FaceRecognizer
)

router = APIRouter()

# Initialize services
detector = ObjectDetector()
classifier = ImageClassifier()
face_recognizer = FaceRecognizer()


@router.get("/")
def home():

    return {
        "message": "Multimodal Computer Vision API Running"
    }


@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...)
):

    # Create uploads directory
    os.makedirs("data/uploads", exist_ok=True)

    # Save uploaded image
    file_path = f"data/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    # Run object detection
    detections, output_path = detector.detect_objects(
        file_path
    )

    output_filename = output_path.split("/")[-1]

    return {
        "filename": file.filename,
        "detections": detections,
        "output_image_url":
            f"http://127.0.0.1:8000/outputs/{output_filename}"
    }


@router.post("/classify")
async def classify_image(
    file: UploadFile = File(...)
):

    # Create uploads directory
    os.makedirs("data/uploads", exist_ok=True)

    # Save uploaded image
    file_path = f"data/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    # Run image classification
    result = classifier.classify_image(file_path)

    return {
        "filename": file.filename,
        "classification": result
    }

@router.post("/recognize-face")
async def recognize_face(
    file: UploadFile = File(...)
):

    # Create uploads directory
    os.makedirs("data/uploads", exist_ok=True)

    # Save uploaded image
    file_path = f"data/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    # Run face recognition
    result, output_path = (
        face_recognizer.recognize_faces(
            file_path
            )
        )
    
    return {
        "filename": file.filename,
        "recognition_result": result,
        "output_image": output_path
    }
