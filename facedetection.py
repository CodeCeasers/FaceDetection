from __future__ import print_function
import cv2 as cv
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

# Define categories for suspicious behavior
suspicious_categories = {
    "suspicious": 0,
    "serious": 0,
    "others": 0
}

# Define thresholds for suspicious behavior
suspicious_threshold = 50
serious_threshold = 100

# Define a list to store the level of suspicious behavior for each frame
suspicious_levels = []

def detectAndDisplay(frame):
    global suspicious_categories

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        
        # Detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

            # Increment the suspicious level based on eye detection
            suspicious_levels.append(1)

    # Check suspicious levels and categorize them
    current_suspicious_level = sum(suspicious_levels)
    if current_suspicious_level >= serious_threshold:
        suspicious_categories["serious"] += 1
    elif current_suspicious_level >= suspicious_threshold:
        suspicious_categories["suspicious"] += 1
    else:
        suspicious_categories["others"] += 1

    # Display frame
    cv.imshow('Capture - Face detection', frame)

# Parse arguments
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

# Load cascades
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier(cv.samples.findFile(face_cascade_name))
eyes_cascade = cv.CascadeClassifier(cv.samples.findFile(eyes_cascade_name))

# Read video stream
camera_device = args.camera
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# Initialize timer
session_duration = 2  # 2 minutes
session_end_time = time.time() + session_duration * 60

# Process frames
while time.time() < session_end_time:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break

# Show results
print("Suspicious Categories:")
print("Suspicious:", suspicious_categories["suspicious"])
print("Serious:", suspicious_categories["serious"])
print("Others:", suspicious_categories["others"])

# Generate graph
categories = list(suspicious_categories.keys())
values = list(suspicious_categories.values())
plt.bar(categories, values, color=['red', 'orange', 'green'])
plt.xlabel('Suspicious Categories')
plt.ylabel('Frequency')
plt.title('Suspicious Behavior during Exam')
plt.savefig('suspicious_behavior.png')  # Save the plotted graph as a PNG
plt.show()

# Release resources
cap.release()
cv.destroyAllWindows()
