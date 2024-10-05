import cv2
import joblib
import numpy as np

# Load the SVM model
model = joblib.load(r'D:\pythonProject\prodigy\task3\svm_mode2.pkl')

# Open the webcam
cap = cv2.VideoCapture(r"C:\Users\Abdullah\Downloads\3326746-hd_1920_1080_24fps.mp4")

# Reduce frame size for better performance (optional)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


# Function to preprocess frame (resizing and normalizing)
def preprocess_frame(frame):
    # Resize the frame to the input size expected by the model (e.g., 120x120)
    resized_frame = cv2.resize(frame, (64, 64))

    # Normalize the frame (pixel values between 0 and 1)
    resized_frame = resized_frame.astype(np.float32) / 255.0

    # Flatten the frame to match the model's input (1D array)
    flattened_frame = resized_frame.flatten().reshape(1, -1)

    return flattened_frame


frame_count = 0  # To control prediction frequency

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Predict on every 10th frame to avoid slowing down
    if frame_count % 10 == 0:
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make prediction
        prediction = model.predict(processed_frame)

        # Map prediction to label
        if prediction==0:
            label="no object"
        elif prediction==2:
            label="dog"
        else:
            label="cat"
    # Draw the label on the frame
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with prediction
    cv2.imshow('Webcam Feed', frame)

    # Increment frame count
    frame_count += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
