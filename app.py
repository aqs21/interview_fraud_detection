from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import threading

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ROBOFLOW_API_KEY = "ATCth3RHKPljJdY3UmHL"
ROBOFLOW_MODEL_ID = "interview-dxisb/3"

detection_result = {}
alert_message = ""
lock = threading.Lock()

# Run Roboflow detection in a separate thread
def detect_fraud_async(frame):
    global detection_result
    resized_frame = cv2.resize(frame, (640, 640))
    _, img_encoded = cv2.imencode('.jpg', resized_frame)

    try:
        response = requests.post(
            f"https://detect.roboflow.com/{'interview-dxisb/3'}",
            files={"file": img_encoded.tobytes()},
            params={"api_key": 'ATCth3RHKPljJdY3UmHL', "confidence": 50, "overlap": 30}
        )
        with lock:
            detection_result = response.json()
    except Exception as e:
        print("Error in object detection:", e)

def generate_frames():
    global alert_message
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            alert_message = "⚠️ No Person Detected (Absence)"
        else:
            alert_message = ""

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Run detection every 10 frames
        if frame_count % 10 == 0:
            threading.Thread(target=detect_fraud_async, args=(frame.copy(),)).start()

        with lock:
            if "predictions" in detection_result:
                for obj in detection_result["predictions"]:
                    confidence = obj["confidence"]
                    width = obj["width"]
                    height = obj["height"]

                    # Filter 1: Confidence must be >= 85%
                    if confidence < 0.70:
                        continue

                    # Filter 2: Ignore tiny objects (area < 5000 px)
                    if width * height < 2000:
                        continue

                    # If passed both filters, show alert
                    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
                    label = f"{obj['class']} ({int(confidence * 100)}%)"
                    x1, y1 = x - w // 2, y - h // 2
                    x2, y2 = x + w // 2, y + h // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    alert_message = f"⚠️ Suspicious Object Detected: {label}"

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def alert_status():
    return jsonify({"message": alert_message})

if __name__ == '__main__':
    app.run(debug=True)
