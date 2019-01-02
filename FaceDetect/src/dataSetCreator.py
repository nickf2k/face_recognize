# import cam as cam
import cv2
import sqlite3
import numpy as np
from PIL import Image

print("Hello")
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()


def insertOrUpdate(Id, name, age, gender):
    connection = sqlite3.connect("database/FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + "'"+str(Id)+"'"
    cursor = connection.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1;
    if isRecordExist == 1:
        cmd = "UPDATE People SET Name =" + "'"+str(name)+"'" + ", Age=" + str(age) + ", Gender=" + "'"+str(gender)+"'" + "WHERE Id=" + "'"+str(Id)+"'"
    else:
        cmd = "INSERT INTO People(Id,Name,Age,Gender) Values(" + "'"+str(Id)+"'" + "," + "'"+str(name)+"'" + "," + str(age) + "," + "'"+str(gender)+"'" + ")"
    connection.execute(cmd)
    connection.commit()
    connection.close()


id = input("Id: ")
name = input("Name: ")
age = input("Age: ")
gender = input("Gender: ")
insertOrUpdate(id, name, age, gender)
sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, 0)
    gray = np.array(gray,dtype=np.uint8)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # incrementing sample number
        sampleNum += 1

        # saving the captured face on the dataset folder
        cv2.imwrite("dataset/User." + id + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow("frame", img)
        #wait for 100ms
    if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    elif sampleNum==50:
            break
cam.release()
cv2.destroyAllWindows()

