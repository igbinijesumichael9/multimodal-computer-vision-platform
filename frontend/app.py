import streamlit as st
import requests

# API ENDPOINTS

DETECTION_API_URL = (
    "http://127.0.0.1:8000/detect"
)

CLASSIFICATION_API_URL = (
    "http://127.0.0.1:8000/classify"
)

FACE_RECOGNITION_API_URL = (
    "http://127.0.0.1:8000/recognize-face"
)

# PAGE CONFIG

st.set_page_config(
    page_title="Multimodal Computer Vision Platform",
    layout="centered"
)

st.title(
    "Multimodal Computer Vision Platform"
)

st.write(
    """
    Upload an image and choose an AI task:
    - Object Detection
    - Image Classification
    - Face Recognition
    """
)

# TASK SELECTION

task = st.selectbox(
    "Select AI Task",
    [
        "Object Detection",
        "Image Classification",
        "Face Recognition"
    ]
)

# FILE UPLOAD

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# PROCESS IMAGE

if uploaded_file is not None:

    # Show uploaded image
    st.subheader("Uploaded Image")

    st.image(
        uploaded_file,
        caption="Input Image",
        use_container_width=True
    )

    # Prepare request file
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    # Run task button
    if st.button("Run AI Task"):

        # OBJECT DETECTION

        if task == "Object Detection":

            response = requests.post(
                DETECTION_API_URL,
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Object Detection Complete"
                )

                st.subheader(
                    "Detection Results"
                )

                # Display detections
                for detection in result["detections"]:

                    st.write(
                        f"{detection['class']} "
                        f"- Confidence: "
                        f"{detection['confidence']}"
                    )

                # Show output image
                st.subheader(
                    "Detected Image"
                )

                st.image(
                    result["output_image"],
                    caption="Detected Objects",
                    use_container_width=True
                )

            else:

                st.error(
                    "Object detection failed."
                )

        # IMAGE CLASSIFICATION

        elif task == "Image Classification":

            response = requests.post(
                CLASSIFICATION_API_URL,
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Image Classification Complete"
                )

                st.subheader(
                    "Classification Result"
                )

                st.write(
                    f"Predicted Class: "
                    f"{result['classification_result']['predicted_class']}"
                )

                st.write(
                    f"Confidence: "
                    f"{result['classification_result']['confidence']}"
                )

            else:

                st.error(
                    "Image classification failed."
                )

        # FACE RECOGNITION

        elif task == "Face Recognition":

            response = requests.post(
                FACE_RECOGNITION_API_URL,
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                recognitions = (
                    result["recognition_result"]
                )

                output_image_path = (
                    result["output_image"]
                )

                st.success(
                    "Face Recognition Complete"
                )

                st.subheader(
                    "Recognition Results"
                )

                # Display all recognized faces
                for idx, recognition in enumerate(
                    recognitions,
                    start=1
                ):

                    st.markdown(
                        f"### Face {idx}"
                    )

                    st.write(
                        f"Person: "
                        f"{recognition['person']}"
                    )

                    st.write(
                        f"Confidence: "
                        f"{recognition['confidence']}"
                    )

                    st.markdown("---")

                # Display annotated image
                st.subheader(
                    "Annotated Image"
                )

                st.image(
                    output_image_path,
                    caption="Recognized Faces",
                    use_container_width=True
                )

            else:

                st.error(
                    "Face recognition failed."
                )