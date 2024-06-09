import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Initialize YOLO model
model = YOLO("best.pt")

# Page title
st.set_page_config(page_title='Coin Classification', page_icon='💸')
st.title('💸 Coin Classification')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to upload images and perform object detection using the YOLO model.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and upload images for object detection.')

    st.markdown('**Under the hood**')
    st.markdown('Libraries used:')
    st.code('''- Ultralytics YOLO for object detection
- PIL for image processing
- Streamlit for user interface
    ''', language='markdown')

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('Upload Images')
    uploaded_images = st.file_uploader("Upload Image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Object detection with YOLO on uploaded images
if uploaded_images:
    for image_file in uploaded_images:
        img = Image.open(image_file)
        img_array = np.array(img)

        # Ensure the image is in the correct format (3 channels)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

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
                st.write("Detected objects:")
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].numpy()
                    class_id = int(box.cls)
                    score = box.conf[0].numpy()
                    class_name = results[0].names[class_id]
                    st.write(f"Class: {class_name}, Score: {score:.2f}, Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {e}")
else:
    st.warning('👈 Upload images to get started!')
