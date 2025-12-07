# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 20:29:58 2025

@author: Eric
"""

from processing import *
from rearranging import *
from projecting import projection_init, projection_show
from perception import perception

import cv2
import numpy as np
from PIL import Image

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
CANVAS_W, CANVAS_H = 1920, 1080

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    )

    window = projection_init()
    cv2.namedWindow("Camera Debug View", cv2.WINDOW_NORMAL)

    print("System Started. Press ESC to exit.")

    while True:
        frame, A, card_map = perception(cap, detector)

        if frame is None:
            continue

        cv2.imshow("Camera Debug View", frame)


        if A is None:
            white = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            projection_show(window, white)
        elif card_map == {}:
            black = np.ones((1080, 1920, 3), dtype=np.uint8) * 5
            projection_show(window, black)
        else:
            if not isinstance(A, np.ndarray):
                A = np.array(A)

            B_paths, B_posis = demap(card_map)

            A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
            C = np.uint8(mid(card_map, A_gray))

            A_img = Image.fromarray(A_gray).convert("L")
            C_img = Image.fromarray(C).convert("L")

            canvas = np.array(compose_canvas(A_img, B_paths, C_img, B_posis))

            if len(canvas.shape) == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            projection_show(window, canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()