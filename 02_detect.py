import cv2
import numpy as np

# define a video capture object
cam = cv2.VideoCapture(0)

# define font face for text
font = cv2.FONT_HERSHEY_SIMPLEX

# define a recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained.yml')

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while(True):
    # Capture the video frame by frame
    ret, frame = cam.read()

    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        id, conf = recognizer.predict(gray[y: y + h, x: x + w])
        username = 'Unknown'

        if(conf < 50):
            if(id is 1):
                username = 'David'

        cv2.putText(frame, username, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Web camera', frame)

    # use 'q' button to close programm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()

# Destroy all the windows
cv2.destroyAllWindows()
