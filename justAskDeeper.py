import os
import glob
from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import caffe
%matplotlib inline

class JustAskDeeper:
    def __init__(self, path=""):
        self.path = path
        self.detector = MTCNN()
        self.faces = []
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        mean_filename='./models/mean.binaryproto'
        proto_data = open(mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean  = caffe.io.blobproto_to_array(a)[0]
        age_net_pretrained='./models/age_net.caffemodel'
        age_net_model_file='./age_net_definitions/deploy.prototxt'
        self.age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
        gender_net_pretrained='./models/gender_net.caffemodel'
        gender_net_model_file='./gender_net_definitions/deploy.prototxt'
        self.gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                            mean=mean,
                            channel_swap=(2,1,0),
                            raw_scale=255,
                            image_dims=(256, 256))
        self.age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        self.gender_list=['Male','Female']

    def set_path(self, path):
        self.path = path
        self.faces.clear()

    def detect_faces(self):
        self.img = plt.imread(self.path)
        self.result = self.detector.detect_faces(self.img)

        for i in range(len(self.result)):
            x1, y1, width, height = self.result[i]['box']
            x1,  y1  =  abs(x1),  abs(y1)
            x2,  y2  =  x1  +  width,  y1  +  height
            keypoints = self.result[i]['keypoints']
            face  =  self.img[y1-50:y2+50,  x1-50:x2+50]
            self.faces.append(face)

    def age_prediction(self):
        for i in range(len(self.result)):
            prediction = self.age_net.predict([self.faces[i]]) 
            print('predicted age:', self.age_list[prediction[0].argmax()])

    def gender_prediction(self):
        for i in range(len(self.result)):
            prediction = self.gender_net.predict([self.faces[i]]) 
            print('predicted gender:', self.gender_list[prediction[0].argmax()])