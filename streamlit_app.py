import streamlit as st
from PIL import Image
from ultralytics import YOLOv10
import cv2
import av
import numpy as np
import queue
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from turn import get_ice_servers
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the YOLOv10 model
model = YOLOv10(f"./model/runs/detect/train/weights/best.pt", task="detect").to(device)
# Queue to handle results between threads
result_queue = queue.Queue()

# Function to handle video frames and perform object detection
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        if box.conf[0].numpy() > 0.7:
            class_name = results[0].names[int(box.cls)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{class_name} {box.conf[0]:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Calculate total and put it in the queue
    total = total_sum(results)
    result_queue.put(total)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Function to process uploaded images
def process_uploaded_images(model, uploaded_images):
    for image_file in uploaded_images:
        img = Image.open(image_file)
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = model(img_array)
        annotated_image = results[0].plot()
        annotated_img_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        st.image(annotated_img_pil, caption=f"Annotated Image: {image_file.name}")
        st.write(f"Total: {total_sum(results)}€" if results[0].boxes else "No objects detected in the image.")

# Function to calculate total sum from detection results
def total_sum(results) -> str:
    total = sum(int(results[0].names[int(box.cls)]) for box in results[0].boxes if box.conf[0].numpy() > 0.7)
    return str(total / 100)

# Main Streamlit app function
def main():
    st.title('💸 Coin Classification')

    with st.expander('About this app'):
        st.markdown('**What can this app do?**')
        st.info('This app captures images from your webcam and performs object detection using the YOLO model.')
        st.markdown('**How to use the app?**')
        st.warning('Allow access to your webcam to start the object detection process.')
        st.markdown('**Under the hood**')
        st.markdown('Libraries used:')
        st.code('''- Ultralytics YOLO for object detection
                - PIL for image processing
                - Streamlit for user interface
                - OpenCV for webcam access
        ''', language='markdown')

    st.sidebar.header('Sources')
    option = st.sidebar.radio('Choose an option:', ['Webcam', 'Upload Images'])

    if option == 'Webcam':
        st.subheader('Webcam Feed')
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": get_ice_servers(),
                "iceTransportPolicy": "relay",
            },
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"frameRate": 10}, "audio": False},
            async_processing=True,
        )
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if not result_queue.empty():
                    total = result_queue.get()
                    labels_placeholder.write(f"Total: {total}€")

    elif option == 'Upload Images':
        st.subheader('Upload Images')
        uploaded_images = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_images:
            process_uploaded_images(model, uploaded_images)

if __name__ == "__main__":
    main()
