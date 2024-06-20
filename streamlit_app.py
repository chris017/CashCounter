import streamlit as st
from PIL import Image
from ultralytics import YOLOv10
import numpy as np
import cv2


def total_sum(results) -> str:
    """Calculate the total sum based on detection results."""
    total = sum(int(results[0].names[int(box.cls)]) for box in results[0].boxes if box.conf[0].numpy() > 0.7)
    return str(total / 100)


def display_webcam_feed(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    st.sidebar.write("Webcam started. Capturing images...")
    placeholder = st.empty()
    total_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = model(img_rgb)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                if box.conf[0].numpy() > 0.7:
                    class_name = results[0].names[int(box.cls)]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            placeholder.image(frame, channels="BGR", use_column_width=True)
            total_placeholder.write(f"Total: {total_sum(results)}€" if results[0].boxes else "")
        except Exception as e:
            st.error(f"Error processing webcam image: {e}")
    cap.release()


def process_uploaded_images(model, uploaded_images):
    for image_file in uploaded_images:
        img = Image.open(image_file)
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        try:
            results = model(img_array)
            annotated_image = results[0].plot()
            annotated_img_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            st.image(annotated_img_pil, caption=f"Annotated Image: {image_file.name}")
            st.write(f"Total: {total_sum(results)}€" if results[0].boxes else "No objects detected in the image.")
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {e}")


def main():
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

    model = YOLOv10("best.torchscript")

    st.sidebar.header('Webcam')
    webcam = st.sidebar.button('Start Webcam')

    st.sidebar.header('Upload Images')
    uploaded_images = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if webcam:
        display_webcam_feed(model)
    else:
        st.warning('👈 Click the button to start the webcam and begin object detection!')

    if uploaded_images:
        process_uploaded_images(model, uploaded_images)
    else:
        st.warning('👈 Upload images to get started!')


if __name__ == "__main__":
    main()
