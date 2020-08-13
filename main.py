import faceDetection
import sys
import matplotlib.pyplot as plt

x = faceDetection.FaceDetection(sys.argv[1])
r = x.face_recognition()
print("ok")
