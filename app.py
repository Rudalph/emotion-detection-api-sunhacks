from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to capture image from webcam and save it to a folder
def capture_image():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Release webcam
    cap.release()
    
    # Save frame to a folder
    img_path = 'images/img.jpg'
    cv2.imwrite(img_path, frame)
    
    return img_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    # Capture image from webcam and save it
    img_path = capture_image()
    
    # Analyze emotion using DeepFace
    try:
        result = DeepFace.analyze(img_path, actions=['emotion'])
        print(result)  # Print the result dictionary
        dominant_emotion = result[0]['dominant_emotion'] 
        print(dominant_emotion)
        return jsonify({'dominant_emotion': dominant_emotion})
    except Exception as e:
        print(e)
        return str(e)

if __name__ == '__main__':
    app.run(port=5001)
