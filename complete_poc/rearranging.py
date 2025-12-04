# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 21:15:48 2025

@author: Eric
"""

from PIL import Image

# -----------------------------
# Canvas
# -----------------------------
CANVAS_W, CANVAS_H = 1920, 1080

A_W, A_H = 960, 540
C_W, C_H = 960, 540
B_W, B_H = 1920, 540

# -----------------------------
# proportional 
# -----------------------------
def resize_to_fit(img, box_w, box_h):
    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

# -----------------------------
# main
# -----------------------------
def compose_canvas(A, B_paths, C, B_posis):
    # white background
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))

    # ===========================
    # A Zone
    # ===========================
    A_r = resize_to_fit(A, A_W, A_H)
    Ax = (A_W - A_r.width) // 2
    Ay = (A_H - A_r.height) // 2
    canvas.paste(A_r, (Ax, Ay))

    # ===========================
    # C Zone
    # ===========================
    C_r = resize_to_fit(C, C_W, C_H)
    Cx = 960 + (C_W - C_r.width) // 2
    Cy = (C_H - C_r.height) // 2
    canvas.paste(C_r, (Cx, Cy))

    # ===========================
    # B Zone
    # ===========================
    for i in range(len(B_posis)):
        Bx, By = B_posis[i]
        B_path = B_paths[i]
        B = Image.open(B_path)
        canvas.paste(B, (Bx, By))

    canvas.save("output_composed.png")

def demap(card_map):
    B_paths = []
    B_posis = []
    for card in card_map:
        Bx, By = card_map[card]
        B_path = card.get_path()
        B_paths.append(B_path)
        B_posis.append((Bx, By))
    return B_paths, B_posis
        
