import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import tempfile

# Configure the application
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Application title
st.title("🔍 AI Object Detection System")

st.write(
    "Upload an image and let YOLOv8 identify "
    "the objects it can recognise."
)

st.info(
    "This application uses a pretrained YOLOv8 "
    "model to detect objects in images."
)

# Load the pretrained model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Detect objects when the button is clicked
    if st.button("Detect Objects", type="primary"):

        try:
            with st.spinner("Detecting objects..."):

                # Save image temporarily for model inference
                with tempfile.TemporaryDirectory() as temp_dir:

                    image_path = Path(temp_dir) / "uploaded_image.jpg"
                    image.save(image_path)

                    model = load_model()

                    results = model.predict(
                        source=str(image_path),
                        conf=0.25,
                        verbose=False
                    )

                    result = results[0]

                    # Get annotated image
                    annotated_image = result.plot()
                    annotated_image = annotated_image[:, :, ::-1]

                # Display the output
                st.subheader("Detection Results")

                st.image(
                    annotated_image,
                    caption="Objects detected by YOLOv8",
                    use_container_width=True
                )

                # Display detected objects
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:

                    st.subheader("Detected Objects")

                    detections = []

                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        object_name = result.names[class_id]

                        detections.append({
                            "Object": object_name,
                            "Confidence": f"{confidence:.2%}"
                        })

                    st.dataframe(
                        detections,
                        use_container_width=True
                    )

                    st.success(
                        f"Detection completed! "
                        f"{len(detections)} object(s) detected."
                    )

                else:
                    st.warning(
                        "No objects were detected. "
                        "Try another image."
                    )

        except Exception as error:
            st.error(
                "An error occurred during detection. "
                "Check your installation and try again."
            )
            st.exception(error)