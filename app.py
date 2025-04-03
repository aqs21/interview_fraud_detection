from flask import Flask, render_template, Response
import cv2
import requests
import threading

app = Flask(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Roboflow API Config
ROBOFLOW_API_KEY = "ATCth3RHKPljJdY3UmHL"
ROBOFLOW_MODEL_ID = "interview_fraud_detection/1"

# Global Variables to store detection results
detection_result = {}
lock = threading.Lock()  # To ensure thread safety

# Function to send frames to Roboflow API (Runs in a separate thread)
def detect_fraud_async(frame):
    global detection_result

    # Resize frame before sending to API (Speeds up processing)
    resized_frame = cv2.resize(frame, (320, 320))

    _, img_encoded = cv2.imencode('.jpg', resized_frame)

    try:
        response = requests.post(
            f"https://detect.roboflow.com/{'interview_fraud_detection/1'}",
            files={"file": img_encoded.tobytes()},
            params={"api_key": 'ATCth3RHKPljJdY3UmHL', "confidence": 50, "overlap": 30}
        )
        with lock:
            detection_result = response.json()
    except Exception as e:
        print("Error in object detection:", e)

# Function to generate frames from webcam
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS for smooth performance

    frame_count = 0  # To run detection every N frames

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw green rectangles for face detection
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Run detection every 10 frames (Reduces lag)
        if frame_count % 10 == 0:
            threading.Thread(target=detect_fraud_async, args=(frame.copy(),)).start()

        # Draw bounding boxes from last detection result
        with lock:
            if "predictions" in detection_result:
                for obj in detection_result["predictions"]:
                    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
                    label = obj["class"]

                    # Draw red box around detected objects
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1  # Increase frame count

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
