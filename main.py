# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 22:25:24 2025

@author: Eric
"""

from processing import *
from rearranging import *

CANVAS_W, CANVAS_H = 1920, 1080

A_W, A_H = 960, 540
C_W, C_H = 960, 540
B_W, B_H = 1920, 540

# registered uuid       
card_indexs = [0, 1, 2, 10]

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


A_path = "A.jpg"

card_map = {vertical_grad: (100, 600),
            horizonal_grad: (130, 800),
            plus_card: (250, 720)}

B_paths, B_posis = demap(card_map)

A = Image.open(A_path).convert("L")  
A = np.array(A)

C = np.uint8(mid(card_map, A))

A_img = Image.fromarray(A).convert("L")   # numpy → Pillow
C_img = Image.fromarray(C).convert("L")   # numpy → Pillow

compose_canvas(A_img, B_paths, C_img, B_posis)
