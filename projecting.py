# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 21:48:19 2025

@author: Eric
"""
import cv2
import numpy as np
from canvas import *
from processing import *
from rearranging import *

CANVAS_W, CANVAS_H = 1920, 1080

A_W, A_H = 960, 540
C_W, C_H = 960, 540
B_W, B_H = 1920, 540

blur3_kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
blur3 = Card(0, "blur3.png", "convolution", blur3_kernel)


vertical_grad_kernel = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
vertical_grad = Card(1, "vertical_grad.png", "convolution", vertical_grad_kernel, "abs")


horizonal_grad_kernel = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])
horizonal_grad = Card(2, "horizonal_grad.png", "convolution", horizonal_grad_kernel, "abs")


plus_card = Card(10, "plus.png", "plus", None)

def projection_init():
    window_name = "HDMI_Projection"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return window_name

window = projection_init()
    
while True:
    A, card_map = conception()
    B_paths, B_posis = demap(card_map)
    
    A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    C = np.uint8(mid(card_map, A_gray))
    
    A_img = Image.fromarray(A_gray).convert("L")   # numpy → Pillow
    C_img = Image.fromarray(C).convert("L")   # numpy → Pillow

    canvas = np.array(compose_canvas(A_img, B_paths, C_img, B_posis))
    
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    projection_show(window, canvas)
    
    if cv2.waitKey(50) & 0xFF == 27:
        break
    