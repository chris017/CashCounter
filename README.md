# Euro Currency Recognizer

Welcome to the Cash Counter app! This Streamlit application is designed to identify and calculate the total value of Euro coins and banknotes from images. It also provides a live webcam feature to classify currency in real-time.

## Features

- **Image Recognition**: Upload images containing Euro coins and banknotes to automatically detect and calculate the total amount.
- **Live Webcam Classification**: Use your webcam to classify Euro currency in real-time.
- **Detailed Analysis**: Get detailed results of detected coins and banknotes with their respective values.
- **Demo Video**: Watch the demo video available in the repository to see the app in action.

## Demo Video

Check out our demo video to see how the app works:

[Demo Video](showcase.mov)

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/chris017/CashCounter.git
    cd euro-currency-recognizer
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the App**:
    ```bash
    streamlit run app.py
    ```

4. **Upload an Image**:
    - Navigate to the app in your browser.
    - Upload an image containing Euro coins and banknotes.
    - View the detected currency and the calculated total amount.

5. **Use Live Webcam Classification**:
    - Select the live webcam option in the app.
    - Allow the app to access your webcam.
    - Hold Euro coins and banknotes in front of the webcam to classify and calculate the total amount in real-time.

## Technologies Used

- **Streamlit**: For building the web app.
- **OpenCV**: For image processing and currency recognition.
- **TensorFlow**: For machine learning and currency classification.


