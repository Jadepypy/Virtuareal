# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 21:48:19 2025

@author: Eric
"""
import cv2
import numpy as np
from processing import *
from rearranging import *

CANVAS_W, CANVAS_H = 1920, 1080

A_W, A_H = 960, 540
C_W, C_H = 960, 540
B_W, B_H = 1920, 540


def projection_init():
    window_name = "HDMI_Projection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    return window_name

def projection_show(window_name, canvas):
    cv2.imshow(window_name, canvas)
    print("Projection updated")