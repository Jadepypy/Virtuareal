import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
CANVAS_W, CANVAS_H = 1920, 1080
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
ANCHOR_IDS = [0, 1, 2, 3]
DEBUG_MODE = True
WINDOW_NAME = "Projector Output"

# Logic Config
VIRTUAL_CARD_OFFSET_Y = 220
PROBE_DISTANCE = 100
MAX_CHAIN_DEPTH = 5
CARD_TTL = 15
INPUT_LINK_TTL = 10

# Directions
LEFT = "LEFT"
RIGHT = "RIGHT"
DOWN = "DOWN"

# --- LOGIC SCRIPTS ---
IMAGE_SCRIPT = """
# Image Logic: Just Render
if self.img_arr is not None:
    self.render_img(self.img_arr)
"""

KERNEL_SCRIPT = """
# --- Helper Definition ---
def render_matrix_visual():
    viz_w, viz_h = self.width, self.height
    k_viz = np.ones((viz_h, viz_w, 3), dtype=np.uint8) * 255
    rows, cols = self.kernel_arr.shape
    color = (0, 0, 0)
    step_x = viz_w / cols
    step_y = viz_h / rows

    for i in range(cols + 1):
        x = int(i * step_x)
        cv2.line(k_viz, (x, 0), (x, viz_h), color, 2)
    for i in range(rows + 1):
        y = int(i * step_y)
        cv2.line(k_viz, (0, y), (viz_w, y), color, 2)

    if rows <= 5 and cols <= 5:
        for r in range(rows):
            for c in range(cols):
                val = self.kernel_arr[r, c]
                text = f"{val:.2f}"
                if abs(val - 1 / 9) < 0.001: text = "1/9"
                elif abs(val - 1.0) < 0.001: text = "1"
                elif abs(val) < 0.001: text = "0"

                center_x = int(c * step_x + step_x / 2)
                center_y = int(r * step_y + step_y / 2)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(k_viz, text, (center_x - text_w // 2, center_y + text_h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.rectangle(k_viz, (0, 0), (viz_w - 1, viz_h - 1), (0, 0, 0), 4)
    self.render_img(k_viz)

# --- Execution ---
# 1. RENDER
render_matrix_visual()

# 2. LOGIC
source = self.resolved_inputs.get(LEFT)
if not source or source.img_arr is None:
    return

# Process
result_arr = source.img_arr
for _ in range(10): 
    result_arr = cv2.filter2D(result_arr, -1, self.kernel_arr)

# Result Creation
# Note: ImageCard will automatically load IMAGE_SCRIPT upon init
result_card = ImageCard(result_arr)

self.project_output(DOWN, result_card)
"""

ADDITION_SCRIPT = """
# --- Helper Definition ---
def render_plus_visual():
    viz = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
    cv2.rectangle(viz, (0,0), (self.width-1, self.height-1), (0,0,0), 4)
    center_x, center_y = self.width//2, self.height//2
    line_len = 40
    cv2.line(viz, (center_x - line_len, center_y), (center_x + line_len, center_y), (0,0,0), 5)
    cv2.line(viz, (center_x, center_y - line_len), (center_x, center_y + line_len), (0,0,0), 5)
    self.render_img(viz)

# --- Execution ---
# 1. RENDER
render_plus_visual()

# 2. LOGIC
k_left = self.resolved_inputs.get(LEFT)
k_right = self.resolved_inputs.get(RIGHT)

if not (k_left and k_right):
    return

try:
    sum_arr = k_left.kernel_arr + k_right.kernel_arr

    # Clean Creation of Virtual Kernel
    # Note: KernelCard will automatically load KERNEL_SCRIPT upon init
    result_card = KernelCard(sum_arr)

    self.project_output(DOWN, result_card)
except Exception:
    return
"""

# --- UTILITIES ---

def load_image_resource(filename, width=200):
    try:
        pil_img = Image.open(filename).convert("RGB")
        w_percent = (width / float(pil_img.size[0]))
        h_size = int((float(pil_img.size[1]) * float(w_percent)))
        pil_img = pil_img.resize((width, h_size), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        fallback = np.zeros((200, 200, 3), dtype=np.uint8)
        fallback[:] = (0, 255, 0)
        return fallback

def project_image_at(canvas, img, top_left):
    if img is None: return
    h, w, _ = img.shape
    x1, y1 = top_left
    x2 = x1 + w
    y2 = y1 + h

    c_y1, c_y2 = max(0, y1), min(CANVAS_H, y2)
    c_x1, c_x2 = max(0, x1), min(CANVAS_W, x2)

    i_y1, i_y2 = c_y1 - y1, h - (y2 - c_y2)
    i_x1, i_x2 = c_x1 - x1, w - (x2 - c_x2)

    if c_x2 > c_x1 and c_y2 > c_y1:
        canvas[c_y1:c_y2, c_x1:c_x2] = img[i_y1:i_y2, i_x1:i_x2]

def get_homography_from_history(anchor_history):
    if len(anchor_history) < 4:
        return None

    src_points = []
    dst_points = []

    ideal_corners = {
        0: (0, 0),
        1: (CANVAS_W, 0),
        2: (CANVAS_W, CANVAS_H),
        3: (0, CANVAS_H)
    }

    for mid, pt in anchor_history.items():
        if mid in ideal_corners:
            src_points.append(pt)
            dst_points.append(ideal_corners[mid])

    if len(src_points) < 4:
        return None

    src = np.array(src_points, dtype=float)
    dst = np.array(dst_points, dtype=float)

    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return M

def transform_point(point, M):
    pt = np.array([[[point[0], point[1]]]], dtype='float32')
    t = cv2.perspectiveTransform(pt, M)
    return int(t[0][0][0]), int(t[0][0][1])