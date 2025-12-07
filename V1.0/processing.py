# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:23:44 2025

@author: Eric
"""
import numpy as np
import time
from scipy.signal import convolve2d
import cv2
#import matplotlib.pyplot as plt

class Card:
    def __init__(self, uuid, path, category, value, value2=None, value3=None, output=None):
        self.uuid = uuid
        self.path = path
        self.category = category
        self.value = value
        self.value2 = value2
        self.value3 = value3
        self.output = output
    
    def hasLeft(self, card_map):
        my_point = card_map[self]
        flag = False
        cards = []
        for card in card_map:
            if card_map[card] in left_area(my_point):
                flag = True
                cards.append(card)
        return flag, cards
    
    
    def get_output(self, card_map, image):
        if self.category == "convolution":
            if self.value2 == None:
                self.output = convolve2d(image, self.value, mode='same', 
                                     boundary='fill', fillvalue=0)
            if self.value2 == "abs":
                self.output = abs(convolve2d(image, self.value, mode='same', 
                                     boundary='fill', fillvalue=0))

        if self.category == "plus":
            flagLeft, cardsLeft = self.hasLeft(card_map)
            if flagLeft:
                for i in range(len(cardsLeft)):
                    if i == 0:
                        result = cardsLeft[i].output
                    else:
                        result += cardsLeft[i].output
                self.output = result
            else:
                self.output = None
    
    def get_path(self):
        return self.path
    
    def append_to_library(self):
        CARD_LIBRARY[self.uuid] = self
    
    def append_to_Apath(self):
        IMAGE_LIBRARY[self.uuid] = self
    
                
                
        

def left_area(point):
    x = point[0]
    y = point[1]
    left_list = []
    for i in range(x-200, x-1):
        for j in range(y-200, y+200):
            left_list.append((i,j))
    return left_list

def right_area(point):
    x = point[0]
    y = point[1]
    left_list = []
    for i in range(x+1, x+200):
        for j in range(y-200, y+200):
            left_list.append((i,j))
    return left_list

def up_area(point):
    x = point[0]
    y = point[1]
    left_list = []
    for i in range(x-200, x+200):
        for j in range(y+1, y+200):
            left_list.append((i,j))
    return left_list

def down_area(point):
    x = point[0]
    y = point[1]
    left_list = []
    for i in range(x-200, x+200):
        for j in range(y-200, y-1):
            left_list.append((i,j))
    return left_list     
            
# registered cards
CARD_LIBRARY = {}
IMAGE_LIBRARY = {}

# define some cards
blur3_kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
blur3 = Card(20, "blur3.png", "convolution", blur3_kernel)
blur3.append_to_library()


vertical_grad_kernel = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
vertical_grad = Card(21, "vertical_grad.png", "convolution", vertical_grad_kernel, "abs")
vertical_grad.append_to_library()


horizonal_grad_kernel = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])
horizonal_grad = Card(22, "horizonal_grad.png", "convolution", horizonal_grad_kernel, "abs")
horizonal_grad.append_to_library()

plus_card = Card(23, "plus.png", "plus", None)
plus_card.append_to_library()


Haruka = Card(10, "haruka.jpg", "image", None)
Haruka.append_to_Apath()

'''
plt.imshow(image, cmap='gray')
'''
'''
card_map = {vertical_grad: (0, 0),
            horizonal_grad: (0, 10),
            plus_card: (15, 10)}

card_map0 = {vertical_grad: (0, 0)}

card_map1 = {horizonal_grad: (0, 0)}
'''

# from left to right
def mid(card_map, image):
    for card, (x, y) in sorted(card_map.items(), key=lambda item: (item[1][0], item[1][1])):
        card.get_output(card_map, image)

    rightmost_card = max(card_map.items(), key=lambda item: item[1][0])
    rc, (x, y) = rightmost_card
    return rc.output


'''
rc = mid(card_map, image)
rc0 = mid(card_map0, image)
rc1 = mid(card_map1, image)

plt.subplot(1, 3, 1)
plt.imshow(rc, cmap='gray')

plt.subplot(1, 3, 2)
plt.imshow(rc0, cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(rc1, cmap='gray')

plt.tight_layout()
plt.show()



fs = 10



while True:
    time.sleep(1/fs)

'''