import cv2, os, uuid, shutil

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

users_dir = 'users'

if not os.path.exists(users_dir):
    os.mkdir(users_dir)

username = input('Enter your username: ')
id = input('Enter id: ')

user_path = os.path.join(users_dir, username)

if os.path.exists(user_path):
    shutil.rmtree(user_path)

os.mkdir(user_path)

sample_num = 0

while(True):
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # saving the captured face in the dataset folder
        path = user_path + '/' + f'Id_{id}.Sample_{sample_num}.jpg'
        print('Fiel saved as: ', path)
        cv2.imwrite(path, gray[y: y + h, x: x + w])

        #incrementing sample number 
        sample_num += 1

    cv2.imshow('Web Camera', frame)

    # wait for 100 miliseconds 
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    # break if the sample number is morethan 20
    elif sample_num > 20:
        break

cam.release()
cv2.destroyAllWindows()