import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

users_dir = 'users'

def get_images_and_labels(path):
    face_samples = []
    ids = []

    for user_dir in os.listdir(path):
        user_full_dir = os.path.join(path, user_dir)

        for user_image in os.listdir(user_full_dir):
            pil_image = Image.open(os.path.join(user_full_dir, user_image)).convert('L')
            image_np = np.array(pil_image, 'uint8')
        
            faces = face_cascade.detectMultiScale(image_np)

            for (x, y, w, h) in faces:
                face_samples.append(image_np[y: y + h, x: x + w])
                ids.append(int(user_image.split(".")[0].replace('Id_', '')))
    
    return face_samples, ids

faces, ids = get_images_and_labels(users_dir)

recognizer.train(faces, np.array(ids))
recognizer.save('trained.yml')
