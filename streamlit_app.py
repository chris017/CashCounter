import streamlit as st
from PIL import Image
from ultralytics import YOLOv10
import numpy as np
import cv2
import time


def totalSum(results) -> str:
    total = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        class_id = int(box.cls)
        score = box.conf[0].numpy()
        if score > 0.7:
            class_name = results[0].names[class_id]
            total += int(class_name)
    total = total / 100
    total = str(total)
    return total

# Initialize YOLO model
model = YOLOv10("best.torchscript")

# Page title
st.set_page_config(page_title='Coin Classification', page_icon='💸')
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

# Initialize webcam
st.sidebar.header('Webcam')
webcam = st.sidebar.button('Start Webcam')

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('Upload Images')
    uploaded_images = st.file_uploader(label="", label_visibility="collapsed",type=["png", "jpg", "jpeg"], accept_multiple_files=True)


if webcam:
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        st.sidebar.write("Webcam started. Capturing images...")

        placeholder = st.empty()
        total_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            # Convert the image to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform object detection
            try:
                results = model(img_rgb)
                
                # Draw bounding boxes on the frame
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].numpy()
                    class_id = int(box.cls)
                    score = box.conf[0].numpy()
                    if score > 0.7:
                        class_name = results[0].names[class_id]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the frame with detections
                placeholder.image(frame, channels="BGR", use_column_width=True)
                # Calculate and display the total
                if len(results[0].boxes) == 0:
                    total_placeholder.write("")
                else:
                    total = totalSum(results)
                    total_placeholder.write(f"Total: {total}€")
                    
            except Exception as e:
                st.error(f"Error processing webcam image: {e}")
                st.write(e)
        cap.release()

else:
    st.warning('👈 Click the button to start the webcam and begin object detection!')

# Object detection with YOLO on uploaded images
if uploaded_images:
    for image_file in uploaded_images:
        img = Image.open(image_file)
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        try:
            results = model(img_array)
            # Plot the results
            annotated_image = results[0].plot()

            # Convert annotated image to display in Streamlit
            annotated_img_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            st.image(annotated_img_pil, caption=f"Annotated Image: {image_file.name}")

            # Display detected objects information
            if len(results[0].boxes) == 0:
                st.write("No objects detected in the image.")
            else:
                total = totalSum(results)
                st.write(f"Total: {total}€")
                
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {e}")
            st.write(e)
else:
    st.warning('👈 Upload images to get started!')
