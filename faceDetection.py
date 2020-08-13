import os
import glob
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

class FaceDetection:
    
    def __init__(self, path):
        self.path = path
        
    def face_recognition(self):
        print("processing " + self.path + "...")
        self.faces = []
        img = plt.imread(self.path)
        detector  =  MTCNN()
        result = detector.detect_faces(img)
        
        for i in result:
            x1, y1, width, height = i['box']
            x2,  y2  =  x1 + width, y1 + height
            face  =  img[y1:y2, x1:x2]
            print(face.shape)
            self.faces.append(face)
            
        return self.faces