import cv2
from PIL import Image
import numpy as np
import os

path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert("L")
        faceNp = np.array(faceImg,np.uint8)
        # split to get ID of image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        print(IDs)
        cv2.imshow("traning", faceNp)
        cv2.waitKey(10)
    return IDs, faces


Ids, faces = getImagesAndLabels(path)
# training
recognizer.train(faces,np.array(Ids))
recognizer.write("recognizer/trainingData.yml")
cv2.destroyAllWindows()
