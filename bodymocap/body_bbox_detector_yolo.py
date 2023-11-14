# Copyright (c) Facebook, Inc. and its affiliates.

import os
import os.path as osp
import sys
import numpy as np
import cv2
import time

import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

class BodyPoseEstimator(object):
    def __init__(self):
        print("Loading Body Pose Estimator")
        self.__load_body_estimator()

    def __load_body_estimator(self):
        self.model = YOLO("yolov8n.pt")
        # self.model.info()
        self.model.to("cuda")
    
    def detect_body_pose(self, img):
        results = self.model.predict(img, imgsz=640, conf=0.5, show=False, classes=0)  # return a list of Results objects
        # print(results)
        result = results[-1]
        bbox_list = []
        for box in result.boxes.cpu().numpy():
            bbox_list.append(box.xywh.tolist()[0])
        return bbox_list
