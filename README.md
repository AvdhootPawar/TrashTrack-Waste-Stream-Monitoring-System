# ♻ TrashTrack-Waste-Stream-Monitoring-System

---

TrashTrack-Waste-Stream-Monitoring-System is an intelligent web-based application meticulously designed to empower **municipalities** in tackling urban trash challenges and fostering **sustainability**. Leveraging advanced object detection technology (YOLOv10), this system provides a practical solution for identifying and categorizing various trash items in real-time, aiding in efficient sorting, recycling efforts, and environmental monitoring. A key feature is its ability for **users to upload their own data**, actively contributing to and enhancing the system's capabilities.

---

## ✨ Core Functionalities

This application offers versatile trash detection capabilities across different media types:

### 1. Image Detection

* **Upload & Analyze:** Allows users to upload image files (JPG, PNG, BMP, WEBP) for instant trash detection.
* **Visual Output:** Displays both the original and the processed image with detected trash items highlighted by bounding boxes and labels.

### 2. Video Detection

* **Process Uploaded Videos:** Supports processing of video files (MP4, AVI, MOV, MKV) to identify trash over time.
* **Frame-by-Frame Analysis:** Applies the YOLOv10 model to each frame, providing a real-time-like detection experience within the video.
* **Downloadable Output:** Generates and allows downloading of the processed video with detected objects.

### 3. Live Webcam Detection

* **Real-time Analysis:** Connects to the user's webcam for live, continuous trash detection.
* **Dynamic Overlay:** Overlays bounding boxes and labels directly onto the live video feed.
* **Interactive Control:** Features a toggle to easily start and stop the webcam feed.

---

## 🔑 Key Features

* **State-of-the-Art Detection:** Powered by the **YOLOv10 object detection model** for high-accuracy trash identification.
* **Configurable Confidence:** Users can adjust the **confidence threshold** to fine-tune detection sensitivity.
* **Intuitive User Interface:** Built with **Streamlit** for an accessible and user-friendly web interface.
* **Robust Video Processing:** Handles video streams efficiently, including temporary file management for seamless playback and download.
* **User Data Contribution:** Enables users to upload their own images/videos, expanding the dataset and improving model performance.
* **Cross-Platform Compatibility:** Designed for easy deployment and use in various environments.

---

## 🛠️ Technical Stack

### Backend

* **Python:** The core programming language.
* **Streamlit:** For creating the interactive web application.
* **PyTorch:** The underlying deep learning framework for the YOLOv10 model.
* **Ultralytics YOLO:** Library providing the YOLOv10 model implementation.
* **OpenCV (`cv2`):** Used for video capture, frame processing, and image manipulation.
* **Pillow (`PIL`):** For image handling.
* **`tempfile` & `os`:** For efficient temporary file management during video processing.

### Frontend

* Streamlit's inherent web interface components provide a clean and responsive design.
* Interactive sliders, radio buttons, file uploaders, and video/image display elements.

### Model & Data

* **YOLOv10 Model:** `best.pt` - a pre-trained or custom-trained YOLOv10 model for trash detection.
* **Roboflow Integration:** (As seen in `Waste.ipynb`) Used for dataset management and downloading, indicating potential custom training data or a specific dataset like TACO-3.

---

## 🌱 Vision & Impact

TrashTrack-Waste-Stream-Monitoring-System aims to be a cornerstone in addressing municipal waste challenges, fostering **cleaner environments**, and promoting a **circular economy**. By providing an accessible and accurate tool for identifying and monitoring waste streams, it can:

* **Assist Municipalities:** Equip local authorities with data-driven insights to manage waste collection, sorting, and disposal more effectively.
* **Promote Sustainability:** Encourage better recycling habits and reduce environmental pollution through informed waste segregation.
* **Empower Citizens:** Enable users to contribute their own data, creating a community-driven initiative that enhances the system's learning and expands its real-world applicability.

This project demonstrates the practical application of cutting-edge AI in building smart cities and achieving crucial environmental goals.

---

## 📸 Screenshots

![Screenshot 2025-07-10 114204](https://github.com/user-attachments/assets/411b361e-601d-4a15-b783-281a51541693)

![Screenshot 2025-07-10 114445](https://github.com/user-attachments/assets/624d5023-ddf1-4090-9692-437aeb04b072)
