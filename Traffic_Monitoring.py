import streamlit as st
import cv2
import cvzone
from ultralytics import YOLO
import math
import datetime
import tempfile
import os
import time  # Importing time for tracking processing duration

# Initialize the YOLO model
model = YOLO("../YOLO-weights/yolov8n.pt")

# Reduced class names for traffic monitoring
classNames = ["car", "motorbike", "bus", "bicycle", "traffic light", "stop sign", "person", "truck"]

# Streamlit App
st.title("Traffic Monitoring Application")
st.write("Upload a video file for real-time traffic monitoring.")

# File uploader
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Create a log file to record detection results
    log_file_path = 'traffic_log.txt'
    with open(log_file_path, 'w') as log_file:
        # Initialize vehicle counts
        vehicle_counts = {name: 0 for name in classNames}

        # Start the timer
        start_time = time.time()

        # Process the video frame by frame
        stframe = st.empty()  # Placeholder for video frame
        stop_button = st.button("Stop")  # Stop button outside the loop

        while True:
            success, img = cap.read()
            if not success:  # Break the loop if no frame is returned
                break

            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                    # Log the detection with timestamp
                    log_file.write(f"{current_time}, {class_name}, {conf}, {x1}, {y1}, {x2}, {y2}\n")

                    # Increment vehicle count
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1

            # Convert the image from BGR to RGB format for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img_rgb, channels="RGB", use_column_width=True)

            # Stop processing if 'Stop' button is pressed
            if stop_button:
                break

    # Release resources
    cap.release()

    # Calculate processing duration
    processing_duration = time.time() - start_time

    # Display vehicle counts
    st.write("Vehicle Counts:")
    for vehicle, count in vehicle_counts.items():
        st.write(f"{vehicle}: {count}")

    # Display total processing time
    st.write(f"Total Processing Time: {processing_duration:.2f} seconds")

    # Provide download link for the log file
    with open(log_file_path, 'rb') as f:
        st.download_button("Download Log File", f, file_name="traffic_log.txt")

    # Cleanup temporary video file
    os.remove(video_path)
