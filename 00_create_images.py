import cv2, os, shutil, json

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

username = input('Enter your username: ')
user_json = 'users.json'
sample_num = 0

def readJson(path):
    with open(path, 'r') as file:
        jsonObject = json.load(file)
        file.close()
        return jsonObject

def getId(path):
    data = readJson(path)
    id = 0

    if len(data.items()) > 0:
        id = int(list(data.items())[-1][0]) + 1

    return id

def writeUser(path, id, username):
    data = readJson(path)
    data[id] = username

    with open(path, 'w') as file:
        json.dump(data, file)
        file.close()

def setup(users_dir, username):
    if not os.path.exists(users_dir):
        os.mkdir(users_dir)

    user_path = os.path.join(users_dir, username)
    if os.path.exists(user_path):
        shutil.rmtree(user_path)

    os.mkdir(user_path)

    if not os.path.exists(user_json):
        file = open(user_json, "w")
        file.write("{}")
        file.close()

    return user_path

user_path = setup('users', username)

id = getId(user_json)

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

writeUser(user_json, id, username)

cam.release()
cv2.destroyAllWindows()