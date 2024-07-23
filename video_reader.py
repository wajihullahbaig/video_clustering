# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:35 2024

@author: Acer
"""
import cv2
from numpy import ndarray

class VideoReader:
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.capture = cv2.VideoCapture(self.file_path)
        self.frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
    def video_meta(self):
        video_meta_data = {
            "fps":self.fps,
            "file_path":self.file_path,
            "frame_width":self.frame_width,
            "frame_height":self.frame_height
            }
        
        return video_meta_data
    def get_frame(self)->ndarray:
        try:
            
            while(True):
                ret,frame = self.capture.read()
                if not ret:
                    break
                yield frame
            self.capture.release()    
        except Exception as ex:
            print(ex)
            self.capture.release()
            return None
        
    def get_nth_frame(self,frame_no)->ndarray:
        try:
            cap = cv2.VideoCapture(self.file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            res, frame = cap.read()
            return frame   
        except Exception as ex:
            print(ex)
            self.capture.release()
            return None