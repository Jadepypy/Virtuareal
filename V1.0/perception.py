# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 20:15:00 2025

@author: Eric
"""
import cv2
import numpy as np
from PIL import Image
from processing import *


# --- CONFIGURATION ---
CANVAS_W, CANVAS_H = 1920, 1080
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# --- LOAD RESOURCES ---
try:
    # Resize to something manageable for real-time convolution (e.g., 200px width)
    # Full 4K convolution is too slow for CPU!
    pil_A = Image.open("A.jpg").convert("L")
    pil_A.thumbnail((200, 200))
    source_img_small = np.array(pil_A)
except:
    print("Warning: A.jpg not found. Using black square.")
    source_img_small = np.zeros((200, 200))

# --- VISION HELPERS ---

def get_homography(corners, ids):
    if ids is None: return None
    found_anchors = {}

    # IDs of the 4 markers on the whiteboard corners
    # Change these if your printed anchors have different IDs
    ANCHOR_IDS = [0, 1, 2, 3]

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in ANCHOR_IDS:
            found_anchors[marker_id] = np.mean(corners[i][0], axis=0)

    if len(found_anchors) < 4: return None

    # Source: Camera Coordinates
    src = np.array([found_anchors[0], found_anchors[1], found_anchors[2], found_anchors[3]], dtype="float32")
    # Dest: Projector Coordinates
    dst = np.array([[0, 0], [CANVAS_W, 0], [CANVAS_W, CANVAS_H], [0, CANVAS_H]], dtype="float32")

    return cv2.getPerspectiveTransform(src, dst)


def transform_point(point, M):
    pt = np.array([[[point[0], point[1]]]], dtype='float32')
    t = cv2.perspectiveTransform(pt, M)
    return int(t[0][0][0]), int(t[0][0][1])

# Force 720p for better performance/resolution balance

ANCHOR_IDS = [100, 101, 102, 103]
INPUT_ID = 20

def perception(cap, detector):
    ret, frame = cap.read()

    corners, ids, rejected = detector.detectMarkers(frame)

    # --- DEBUG DRAWING (On Camera View) ---
    # 1. Draw Green squares around all detected cards
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # 2. Highlight Anchors in Blue to confirm lock
    ANCHOR_IDS = [0, 1, 2, 3]
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in ANCHOR_IDS:
                c = np.mean(corners[i][0], axis=0).astype(int)
                cv2.circle(frame, tuple(c), 15, (255, 0, 0), 3)  # Blue circle
                cv2.putText(frame, "ANCHOR", (c[0] + 20, c[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 3. Calibration
    M = get_homography(corners, ids)
    
    card_map = {}
    A = None
    
    # --- STEP 1: LOCATE CARDS ---
    if ids is None or M is None:
        return frame, None, {}
    else:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in CARD_LIBRARY:
                card = CARD_LIBRARY[marker_id]
                
                # Get center in Camera Space
                cam_center = np.mean(corners[i][0], axis=0)
                
                # Convert to Projector Space
                cx, cy = transform_point(cam_center, M)
                card_map[card] = (cx, cy)
            if marker_id in IMAGE_LIBRARY:
                img_card = IMAGE_LIBRARY[marker_id]
                img_path = img_card.path
                A = np.array(Image.open(img_path).convert("RGB"))
    return frame, A, card_map