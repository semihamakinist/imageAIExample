# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:23:57 2019

@author: semiha
"""

"""
python singleImagePrediction.py -i test-images\1_1.jpg 
                                -m models\resnet50_weights_tf_dim_ordering_tf_kernels.h5 
                                --mod normal --type stream

"""
import imageai_units as unt
import time as tm
import argparse
#import os


if __name__ == "__main__":
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-m", "--model", required=True,
                    help="base path to deep learning model directory")
    ap.add_argument("--mod", default="normal",
                    help="normal, fast, faster, fastest")
    ap.add_argument("--type", default="stream",
                    help="input type can be a file, array and stream")
    args = vars(ap.parse_args())
    
    #set system path
    model_path = args["model"]
    img_path = args["image"]
    mod = args["mod"]
    inType = args["type"] 

    #set model
    prediction = unt.predictionModelLoad(model_path, mod)
    start_time = tm.time()
    predictions, probabilities = prediction.predictImage(img_path,
                                                         result_count=1,
                                                         input_type=inType)
    print("mod: {} - time: {} sn".format(mod, (tm.time() - start_time)))
    
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print("predict_label: {} - predict_score: {}".format(eachPrediction, eachProbability))