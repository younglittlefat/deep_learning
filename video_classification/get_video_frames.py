#encoding=utf-8
import sys
import os
import shutil
import random
from PIL import Image

import numpy as np
import cv2 as cv

video_dir = "./video"
output_dir = "./frames"

def get_video_frames_on_avg():
    total_frame_num = 30
    video_files = os.listdir(video_dir)
    for filename in video_files:
        filepath = os.path.join(video_dir, filename)
        cap = cv.VideoCapture(filepath)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        frames_num = cap.get(7)
        print "name: %s    total frame num: %d" % (filename, frames_num)
        output_path = os.path.join(output_dir, filename)
        os.mkdir(output_path)
        frame_list = []
        now_idx = 0
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frame_list.append((now_idx, frame))
            now_idx += 1
            #cv.imshow("img", frame)
            #cv.waitKey(10)

        random.shuffle(frame_list)
        for i in range(total_frame_num):
            cv.imwrite(os.path.join(output_path, str(frame_list[i][0])+".jpg"), frame_list[i][1], [int(cv.IMWRITE_JPEG_QUALITY), 100])
        
        cap.release()


get_video_frames_on_avg()
