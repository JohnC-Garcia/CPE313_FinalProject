import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(page_title="Fast Product Detection Demo")

st.title("🛒 Fast Product Detection Preview")
st.write("Disclaimer: This demo simulates detection only for the first few frames if the file you uploaded is a video.")

@st.cache_resource
def load_model():
    return RTDETR("weights.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
if uploaded_file:
    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image")
        with st.spinner("Detecting..."):
            result = model(img)
            st.image(result[0].plot(), caption="Detected Products")

    elif uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
        st.write("Processing video (only 5 frames)...")

        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        preview_pairs = []

        while cap.isOpened() and len(preview_pairs) < 5:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = model(frame_rgb)
                annotated = result[0].plot()
                preview_pairs.append((frame_rgb, annotated))
            frame_count += 1

        cap.release()

        st.subheader("🔍 Detection Preview: Original vs. Labeled")
        for i, (original, labeled) in enumerate(preview_pairs):
            st.markdown(f"**Frame {i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original", use_column_width=True)
            with col2:
                st.image(labeled, caption="With Labels", use_column_width=True)
