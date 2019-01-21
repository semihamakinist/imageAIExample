# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:49:50 2019

@author: semiha
"""

"""
python liveCameraObjectDetection.py -u ipCameraURL 
                                    -m models\resnet50_coco_best_v2.0.1.h5 
                                    -o camera_detected_video 
                                    --mod normal --type stream
"""

import imageai_units as unt
import argparse
import cv2  

def forFrame(frame_number, output_array, output_count, detected_frame):
#    print("FOR FRAME " , frame_number)
#    print("Output for each object : ", output_array)
#    print("Output count for unique objects : ", output_count)
#    print("Returned Objects is : ", type(detected_frame))
#    print("------------END OF A FRAME --------------")
    
#    Processing Frame :  8
#    FOR FRAME  8
#    Output for each object :  [{'name': 'car', 'percentage_probability': 82.96539783477783, 'box_points': array([392,  23, 458,  49])}, {'name': 'car', 'percentage_probability': 96.66531085968018, 'box_points': array([268,  69, 386, 172])}]
#    Output count for unique objects :  {'car': 2}
#    Returned Objects is :  <class 'numpy.ndarray'>
    
    cv2.imshow('IP Camera', detected_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--ipUrl", required=True,
                    help="path to input image")
    ap.add_argument("-m", "--model", required=True,
                    help="base path to deep learning model directory")
    ap.add_argument("-o", "--output", required=True,
                    help="base path to deep learning model directory")
    ap.add_argument("--mod", default="normal",
                    help="normal, fast, faster, fastest and flash")
    ap.add_argument("--type", default="stream",
                    help="input type can be a file, array and stream")
    args = vars(ap.parse_args())
    
    #set system path
    model_path = args["model"]
    ipCameraUrl = args["ipUrl"]
    output_path = args["output"]
    mod = args["mod"]
    inType = args["type"] 
    
    camera = cv2.VideoCapture(ipCameraUrl)
    detector = unt.detectionModelLoad(model_path)
    
    #set SPECIFIC object
    custom_objects = detector.CustomObjects(person=True, bicycle=True,
                                            motorcycle=True, car=True)
    
    detector.detectObjectsFromVideo(camera_input=camera,
                                    output_file_path=output_path,
                                    frames_per_second=10,
                                    per_frame_function=forFrame,
                                    log_progress=True,
                                    minimum_percentage_probability=40,
                                    return_detected_frame=True)