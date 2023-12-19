from flask import Flask, render_template, Response, redirect, url_for
from pose_detection import pose_detection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/options')
def options():
    return render_template('options.html')

@app.route('/video_feed')
def video_feed():
    return Response(pose_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_detection')
def pose_detection_page():
    return render_template('index.html')

@app.route('/redirect_to_options')
def redirect_to_options():
    return redirect(url_for('options'))

if __name__ == '__main__':
    app.run(debug=True)
 
 # @app.route('/emotion_detection')
# def emotion_detection_page():
#     return render_template('emotion_detection.html')   

# emotion_model = load_emotion_model()

# # Video capture
# cap = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         success, frame = cap.read()  # Read the frame from the camera

#         # Perform emotion detection on the frame
#         emotion = detect_emotion(frame)

#         # Draw the emotion text on the frame
#         frame = draw_text(frame, emotion)

#         # Convert the frame to JPEG
#         ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def detect_emotion(frame):
#     # Preprocess input frame
#     preprocessed_frame = preprocess_input(frame)

#     # Perform emotion detection
#     emotion_prediction = emotion_model.predict(np.expand_dims(preprocessed_frame, axis=0))
#     emotion_label = np.argmax(emotion_prediction)

#     # Map the label to emotion
#     emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#     detected_emotion = emotions[emotion_label]

#     return detected_emotion

# def draw_text(frame, text):
#     cv2.putText(frame, f'Emotion: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     return frame
    