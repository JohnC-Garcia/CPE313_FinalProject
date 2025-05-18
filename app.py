import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import base64

#Page title
st.set_page_config(page_title="Product Detector")

#Title
st.title("🛒 Product Detection on Store Shelves")
st.write("Upload an image or video of a store shelf to detect products using a fine-tuned RTDETR-L model.")

#Load model
@st.cache_resource
def load_model():
    model = RTDETR("weights.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image or a video file...", type=["jpg", "jpeg", "png", "mp4"])
if uploaded_file:
    file_type = uploaded_file.type

    #If image
    if file_type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting products in image..."):
            results = model(image)
            annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detected Products", use_column_width=True)

    #If video
    elif file_type == "video/mp4":
        st.video(uploaded_file, format="video/mp4")
        st.write("Processing video, this may take a moment...")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(tempfile.gettempdir(), "detected_video.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        max_preview_frames = 5
        preview_images = []

        with st.spinner("Running detection on video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(img_rgb)

                # Get annotated RGB image
                annotated_rgb = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                # Write to output video
                out.write(annotated_frame)

                # Store preview images
                if frame_count < max_preview_frames:
                    preview_images.append(annotated_rgb)
                frame_count += 1

        cap.release()
        out.release()

        #Show preview
        st.subheader("🔍 Detection Preview (First few frames)")
        for i, preview in enumerate(preview_images):
            st.image(preview, caption=f"Frame {i + 1}", use_column_width=True)

        #Show processed video
        st.subheader("🎞️ Full Annotated Video")
        st.video(output_path)

        #Download link
        with open(output_path, "rb") as file:
            video_data = file.read()
            b64 = base64.b64encode(video_data).decode()
            href = f'<a href="data:video/mp4;base64,{b64}" download="annotated_video.mp4">📥 Download Annotated Video</a>'
            st.markdown(href, unsafe_allow_html=True)
