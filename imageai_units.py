# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:44:17 2019

@author: semiha
"""

from imageai.Detection import VideoObjectDetection
from imageai.Prediction import ImagePrediction

import os

def predictionModelLoad(model_path, mod="normal"):
    multiple_prediction = ImagePrediction()
    multiple_prediction.setModelTypeAsResNet()
    multiple_prediction.setModelPath(model_path)
    multiple_prediction.loadModel(prediction_speed=mod)
    
    return multiple_prediction

def detectionModelLoad(model_path, mod="normal"):
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel(detection_speed=mod)
    
    return detector
    

def addArray(folderPath):
    images_arrays = []
    images_names = []
    
    for each_file in os.listdir(folderPath):
        if(each_file.endswith(".jpg") or each_file.endswith(".png")):
            images_arrays.append(os.path.join(folderPath, each_file))
            images_names.append(each_file)
    return (images_arrays, images_names)