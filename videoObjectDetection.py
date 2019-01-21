# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:23:02 2019

@author: semiha
"""

"""
python videoObjectDetection.py -v videos\traffic.mp4
                               -m models\resnet50_coco_best_v2.0.1.h5
                               --mod normal --type stream
"""
import imageai_units as unt
import argparse


if __name__ == "__main__":
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
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
    video_path = args["video"]
    output_path = args["output"]
    mod = args["mod"]
    inType = args["type"] 
    

    detector = unt.detectionModelLoad(model_path)
    
#    detectObjectsFromVideo(self, input_file_path="", camera_input = None, 
#                           output_file_path="", frames_per_second=20, 
#                           frame_detection_interval=1, minimum_percentage_probability=50,
#                           log_progress=False, display_percentage_probability=True, 
#                           display_object_name = True, save_detected_video = True,
#                           per_frame_function = None, per_second_function = None, 
#                           per_minute_function = None, video_complete_function = None, 
#                           return_detected_frame = False )
    
    video_path = detector.detectObjectsFromVideo(input_file_path=video_path,
                                                 output_file_path=output_path,
                                                 frames_per_second=20, 
                                                 log_progress=True)
    print(video_path)