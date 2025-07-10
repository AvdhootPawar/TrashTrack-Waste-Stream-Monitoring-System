# --- START OF FILE App.py ---

import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
from pathlib import Path
import numpy as np
from io import BytesIO
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLOv10 Trash Detection",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title("♻ Trash Detection using YOLOv10")
st.markdown("Detect trash items in images, videos, or live webcam feed.")
st.markdown("---") # Divider

# --- Sidebar Configuration ---
st.sidebar.header("⚙ Configuration")

# Model configuration
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    help="Adjust the confidence threshold for object detection."
)

# Source type selection
source_options = ["Image", "Video", "Webcam"]
source_type = st.sidebar.radio("Select Source Type", source_options, index=0)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv10 model from the specified path."""
    try:
        model = YOLO('best.pt')
        # st.sidebar.success("Model loaded successfully!") # Optional: keep sidebar clean
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_yolo_model()

# --- Video Processing Function (Keep as previously defined) ---
def process_uploaded_video(video_bytes, confidence_threshold):
    # ... (Keep the robust video processing function from the previous answer) ...
    video_path = None
    output_path = None
    processed_video_bytes = None
    cap = None
    out = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_in:
            tfile_in.write(video_bytes)
            video_path = tfile_in.name

        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile_out.name
        tfile_out.close()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = -1

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             st.warning("Trying 'mp4v' codec...")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
             if not out.isOpened(): raise IOError(f"Could not open video writer. Output: {output_path}")

        if model is None: raise ValueError("YOLO Model is not loaded.")

        st.info(f"Processing video... (Frames: {total_frames if total_frames > 0 else 'Unknown'})")
        progress_text = "Processing frames..."
        progress_bar = st.progress(0, text=progress_text)
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            results = model(frame, conf=confidence_threshold, verbose=False)
            processed_frame = results[0].plot()
            out.write(processed_frame)
            frame_count += 1
            if total_frames > 0:
                progress = min(1.0, frame_count / total_frames)
                progress_bar.progress(progress, text=f"{progress_text} {frame_count}/{total_frames}")
            else:
                 progress_bar.progress(0.0, text=f"{progress_text} Frame {frame_count}")

        if cap: cap.release()
        if out: out.release()
        end_time = time.time()
        st.info(f"Video processing took {end_time - start_time:.2f} seconds.")
        with open(output_path, 'rb') as f: processed_video_bytes = f.read()
        st.success("✅ Video processing complete!")

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        if cap and cap.isOpened(): cap.release()
        if out and out.isOpened(): out.release()
        processed_video_bytes = None
    finally:
        if video_path and os.path.exists(video_path):
            try: os.unlink(video_path)
            except Exception as del_e: st.warning(f"Could not delete temp input {os.path.basename(video_path)}: {del_e}")
        if output_path and os.path.exists(output_path):
            try: os.unlink(output_path)
            except Exception as del_e: st.warning(f"Could not delete temp output {os.path.basename(output_path)}: {del_e}")
    return processed_video_bytes


# --- Main Application Logic ---

if model is not None: # Only proceed if the model loaded successfully
    # --- Handle Image Input ---
    if source_type == "Image":
        st.header("🖼 Image Detection")
        uploaded_file = st.file_uploader(
            "Upload an image file", type=['jpg', 'jpeg', 'png', "bmp", "webp"], key="image_uploader"
        )
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                try:
                    uploaded_image = Image.open(uploaded_file)
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
                    uploaded_image = None
            with col2:
                st.subheader("Detected Image")
                if st.button('✨ Detect Trash in Image', key="detect_image_button") and uploaded_image:
                    try:
                        with st.spinner("Detecting objects..."):
                            results = model(source=uploaded_image, conf=confidence)
                            res_plotted = results[0].plot()[:, :, ::-1] # BGR to RGB
                            st.image(res_plotted, caption='Detected Image', use_container_width=True)
                    except Exception as ex:
                        st.error("An error occurred during image detection.")
                        st.exception(ex)
                elif not uploaded_image:
                     st.warning("Please upload a valid image first.")
                else:
                    st.info("Click the 'Detect' button to see the results.")
        else:
            st.info("👆 Upload an image using the browser above.")

    # --- Handle Video Input ---
    elif source_type == "Video":
        st.header("🎬 Video Detection")
        uploaded_file = st.file_uploader(
            "Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'], key="video_uploader"
        )
        if uploaded_file is not None:
            video_bytes = uploaded_file.read()
            st.subheader("Original Video")
            col1_orig, col2_orig, col3_orig = st.columns([1, 3, 1])
            with col2_orig: st.video(video_bytes)
            st.divider()
            if st.button('✨ Detect Trash in Video', key="detect_video_button"):
                with st.spinner('Processing video...'):
                    processed_video_bytes = process_uploaded_video(video_bytes, confidence)
                if processed_video_bytes:
                    st.subheader("Detected Video")
                    col1_proc, col2_proc, col3_proc = st.columns([1, 3, 1])
                    with col2_proc:
                        st.video(processed_video_bytes, format="video/mp4")
                        st.download_button(label="⬇ Download Processed Video", data=processed_video_bytes, file_name=f"processed_{uploaded_file.name}", mime="video/mp4")
        else:
            st.info("👆 Upload a video using the browser above.")

    # --- Handle Webcam Input ---
    elif source_type == "Webcam":
        st.header("📷 Live Webcam Detection")

        # Initialize session state
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'cap' not in st.session_state:
             st.session_state.cap = None # Store camera object in session state

        # Use a toggle button for better UX
        start_stop_toggle = st.toggle("Start/Stop Webcam", value=st.session_state.webcam_active, key="webcam_toggle")

        # Update session state based on toggle
        st.session_state.webcam_active = start_stop_toggle

        frame_placeholder = st.empty() # Placeholder to display frames

        if st.session_state.webcam_active:
            # --- Start Webcam ---
            if st.session_state.cap is None: # Only create a new capture object if needed
                 st.session_state.cap = cv2.VideoCapture(0) # Use camera 0 (default)
                 # Add small delay to allow camera init? Optional.
                 # time.sleep(0.5)
                 if not st.session_state.cap.isOpened():
                      st.error("Error: Cannot open webcam. Please check permissions and connections.")
                      st.session_state.webcam_active = False # Turn off state
                      st.session_state.cap = None # Reset cap object
                      # Force rerun to update the toggle button state visually
                      if start_stop_toggle: # If toggle was ON but camera failed
                          st.experimental_rerun()


            # --- Processing Loop (if active and camera opened) ---
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                 st.info("Webcam active. Detecting trash...")
                 while st.session_state.webcam_active: # Loop while toggle is ON
                      ret, frame = st.session_state.cap.read()
                      if not ret:
                           st.warning("Failed to grab frame from webcam. Stopping.")
                           st.session_state.webcam_active = False # Turn off state
                           if st.session_state.cap:
                                st.session_state.cap.release() # Release camera
                                st.session_state.cap = None
                           st.experimental_rerun() # Rerun to update UI
                           break # Exit loop

                      # --- Perform Inference ---
                      results = model(frame, conf=confidence, verbose=False) # verbose=False is good here
                      processed_frame = results[0].plot() # Get annotated frame (BGR)

                      # --- Display Frame ---
                      # Convert BGR (OpenCV default) to RGB (Streamlit expects)
                      processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                      frame_placeholder.image(processed_frame_rgb, caption="Live Webcam Feed (Processed)", use_container_width=True)

                      # Check if toggle was turned off during processing (important!)
                      # Re-read the toggle state within the loop is tricky with st.toggle's direct binding to session_state.
                      # The 'while st.session_state.webcam_active:' condition handles this.
                      # If the user clicks the toggle, st re-runs, webcam_active becomes False, and the loop condition fails on the next iteration.


        # --- Stop Webcam (executed when webcam_active becomes False) ---
        if not st.session_state.webcam_active and st.session_state.cap is not None:
            st.info("Stopping webcam...")
            st.session_state.cap.release()
            st.session_state.cap = None
            frame_placeholder.empty() # Clear the last frame
            st.info("Webcam stopped.")
            # Optional: Rerun one last time to ensure the "Webcam stopped" message persists correctly
            # st.experimental_rerun()

        # Initial message when webcam is off
        elif not st.session_state.webcam_active and st.session_state.cap is None:
             frame_placeholder.info("Webcam is off. Toggle the switch above to start.")


else: # Model did not load
    st.error("🚨 Model failed to load! Please check the model path (best.pt) and environment.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Trash Detection App | YOLOv10")
# --- END OF FILE App.py ---