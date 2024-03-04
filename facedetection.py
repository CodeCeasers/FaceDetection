import cv2
import time
import matplotlib.pyplot as plt

def detect_cheating():
    # loding the face detections models

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Open camera
    cap = cv2.VideoCapture(0)

    # Initialize variables
    start_time = time.time()
    cheating_count = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Interview Camera', frame)

        # Convert frame to grayscale for better processing speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within each face region
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Draw rectangle around face region
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Check if no eyes are detected within a face region
            if len(eyes) == 0:
                cheating_count += 1

        # Count frames and check for cheating
        if time.time() - start_time > 3:  # Check every 3 seconds
            start_time = time.time()
            total_frames += 1

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate cheating percentage
    if total_frames > 0:
        cheating_percentage = (cheating_count / total_frames) * 100
        print("Cheating Percentage:", cheating_percentage)
        # Generate bar chart
        labels = ['Cheating', 'Not Cheating']
        values = [cheating_percentage, 100 - cheating_percentage]
        plt.bar(labels, values, color=['red', 'green'])
        plt.xlabel('Status')
        plt.ylabel('Percentage')
        plt.title('Applicant Cheating Detection')
        plt.show()
        # Decision based on threshold
        if cheating_percentage > 20:  # You can adjust this threshold based on your requirements
            print("Applicant is likely cheating.")
        else:
            print("Applicant is not cheating.")
    else:
        print("No frames captured.")

# Run the function to detect cheating
detect_cheating()
