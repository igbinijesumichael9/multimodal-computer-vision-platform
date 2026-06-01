# Building a Multimodal Computer Vision Platform with FastAPI, Streamlit, YOLOv8, ResNet50, and Docker

## Introduction

Computer vision applications are often built as standalone solutions that focus on a single task such as object detection, image classification, or facial recognition. In real-world scenarios, organizations frequently require multiple computer vision capabilities within a unified platform.

To address this challenge, I developed a Multimodal Computer Vision Platform that integrates object detection, image classification, and face recognition into a single application. The platform provides an intuitive web interface, RESTful APIs, and containerized deployment using Docker.

---

## Problem Statement

Many computer vision solutions are designed to solve a specific problem but lack flexibility for broader image analysis tasks. Users often need to switch between multiple tools and workflows to perform different visual recognition tasks.

The goal of this project was to build a unified system capable of:

* Detecting objects in images
* Classifying image content
* Recognizing human faces
* Providing results through a simple web interface
* Exposing functionality through reusable APIs
* Supporting portable deployment through containerization

---

## Solution Overview

The solution combines multiple deep learning models and computer vision techniques into a single platform.

The application consists of:

* Streamlit frontend for user interaction
* FastAPI backend for API services
* YOLOv8 for object detection
* ResNet50 for image classification
* Dlib-based face recognition
* Docker for deployment and portability

Users upload an image, select an AI task, and receive inference results along with annotated output images when applicable.

---

## System Architecture

The platform follows a modular architecture that separates user interaction, API services, and machine learning inference components.

Workflow:

1. User uploads an image through the Streamlit interface.
2. The frontend sends the image to the FastAPI backend.
3. The backend routes the request to the appropriate AI model.
4. The model performs inference.
5. Results are returned as structured JSON responses.
6. The frontend displays predictions and output images.

---

## Technologies Used

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Machine Learning

* PyTorch
* TorchVision
* YOLOv8 (Ultralytics)
* ResNet50
* Dlib
* Face Recognition

### Deployment

* Docker

---

## Model Selection

### Object Detection

YOLOv8 was selected because of its balance between speed and accuracy. It provides real-time object detection while maintaining strong performance across diverse object categories.

### Image Classification

ResNet50 was chosen due to its proven effectiveness on large-scale image classification tasks and availability of pretrained ImageNet weights.

### Face Recognition

The Dlib-based face recognition pipeline was implemented to detect and compare facial embeddings against known identities.

---

## Key Challenges and Solutions

### Challenge 1: Docker Dependency Installation

The face recognition component depends on Dlib, which requires native compilation tools and system libraries.

Solution:

* Installed required build dependencies within the Docker image.
* Configured CMake and supporting libraries for successful compilation.

### Challenge 2: Model Download Issues

The pretrained ResNet50 weights generated hash validation errors during container startup.

Solution:

* Added a dedicated preload script to download model weights during the Docker build stage.
* Ensured models were cached before runtime.

### Challenge 3: API Startup Timing

The frontend attempted to connect before the backend was fully initialized.

Solution:

* Modified container startup sequencing to allow the backend sufficient initialization time before launching the frontend.

---

## Results

The platform successfully performs:

### Object Detection

* Detects multiple objects in uploaded images.
* Returns confidence scores.
* Generates annotated output images.

### Image Classification

* Predicts image classes using pretrained deep learning models.
* Returns confidence probabilities.

### Face Recognition

* Identifies known individuals from uploaded images.
* Generates annotated recognition outputs.

The entire application is containerized and can be deployed consistently across different environments.

---

## Skills Demonstrated

This project demonstrates practical experience in:

* Machine Learning Engineering
* Computer Vision
* Deep Learning
* FastAPI Development
* REST API Design
* Docker Containerization
* Model Deployment
* Frontend Integration
* Software Engineering
* Debugging and Troubleshooting

---

## Future Improvements

Potential enhancements include:

* Real-time webcam inference
* Video object detection and tracking
* OCR and document understanding
* Face enrollment through the user interface
* Cloud deployment on AWS, Azure, or GCP
* Continuous Integration and Continuous Deployment (CI/CD)
* Monitoring and observability dashboards

---

## Conclusion

This project demonstrates how multiple computer vision capabilities can be integrated into a unified, production-ready platform. By combining modern deep learning models, API-driven architecture, and containerized deployment, the solution provides a scalable foundation for future computer vision applications and showcases practical machine learning engineering skills applicable to real-world AI systems.
