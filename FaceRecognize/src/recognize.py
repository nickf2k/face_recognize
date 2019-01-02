import sqlite3
import cv2
import numpy as np
import pickle
from PIL import Image

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")
id = 0
# set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (203, 0, 52)


# get data from database
def getProfile(id):
    connection = sqlite3.connect("database/FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + "'" + str(id) + "'"
    cursor = connection.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    connection.close()
    return profile



while True:
    ret, img = cam.read()
    # newIMg = img.convert("L")
    if img is not None:
        gray = cv2.cvtColor(img,7)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print(rec.getLables())
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        profile = getProfile(id)
        if profile!= None:
            #set text to window

            cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), fontface, fontScale, fontColor, 2)
            cv2.putText(img, "Age: " + str(profile[2]), (x, y + h + 60), fontface, fontScale, fontColor, 2)
            cv2.putText(img, "Gender: " + str(profile[3]), (x, y + h + 90), fontface, fontScale, fontColor, 2)
        cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
