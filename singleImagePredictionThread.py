# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:19:12 2019

@author: semiha
"""

"""
python singleImagePredictionThread.py -f test\ 
                                      -m models\resnet50_weights_tf_dim_ordering_tf_kernels.h5 
                                      --mod normal --type file

"""
import imageai_units as unt
import threading
import time as tm
import argparse
import os

class PredictionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    
    def init_set(self, model_path, imgFolderPath, mod, inType):    
#        threading.Thread.init(self)
        
        self.imgFolderPath = imgFolderPath
        self.model_path = model_path
        self.inType = inType
        self.mod = mod
        
    def run(self):  
        self.prediction = unt.predictionModelLoad(self.model_path, self.mod)
        
        for eachPicture in os.listdir(self.imgFolderPath):
            if eachPicture.endswith(".png") or eachPicture.endswith(".jpg"):
                imgPath = os.path.join(self.imgFolderPath, eachPicture)
                
                start_time = tm.time()
                predictions, probabilities = self.prediction.predictImage(imgPath,
                                                                          result_count=1,
                                                                          input_type=self.inType)
                print("mod: {} - time: {} sn".format(mod, (tm.time() - start_time)))
                
                for prediction, probability in zip(predictions, probabilities):
                    print(prediction , " : " , probability)


if __name__ == "__main__":
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--imageFolder", required=True,
                    help="path to input image")
    ap.add_argument("-m", "--model", required=True,
                    help="base path to deep learning model directory")
    ap.add_argument("--mod", default="normal",
                    help="normal, fast, faster, fastest")
    ap.add_argument("--type", default="stream",
                    help="input type can be a file, array and stream")
    args = vars(ap.parse_args())
    
    #set system path
    modelPath = args["model"]
    imgFolderPath = args["imageFolder"]
    mod = args["mod"]
    inType = args["type"] 

    #set model
    predictionThread = PredictionThread()
    predictionThread.init_set(modelPath, imgFolderPath, mod, inType)
    predictionThread.start()
