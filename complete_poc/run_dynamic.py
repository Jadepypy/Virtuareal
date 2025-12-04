import cv2
import numpy as np
from PIL import Image
import processing

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

# --- DEFINE CARDS ---
blur_k = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
edge_k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# ID 0, 1, 2, 3 are reserved for corner anchors
CARD_LIBRARY = {
    # ID 10: The "Source" Card (Image A)
    10: processing.Card(10, "Source Image", type="source", source_image=source_img_small),

    # ID 0: Blur Card
    20: processing.Card(20, "Blur Tool", type="kernel", kernel=blur_k),

    # ID 1: Edge Detect Card
    # 30: processing.Card(1, "Edge Tool", type="kernel", kernel=edge_k, op_mode="abs"),
}


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


# --- MAIN LOOP ---

def main():
    cap = cv2.VideoCapture(0)
    # Force 720p for better performance/resolution balance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE))

    print("System Started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        corners, ids, rejected = detector.detectMarkers(frame)

        # --- DEBUG DRAWING (On Camera View) ---
        # 1. Draw Green squares around all detected cards
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 2. Highlight Anchors in Blue to confirm lock
        ANCHOR_IDS = [100, 101, 102, 103]
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in ANCHOR_IDS:
                    c = np.mean(corners[i][0], axis=0).astype(int)
                    cv2.circle(frame, tuple(c), 15, (255, 0, 0), 3)  # Blue circle
                    cv2.putText(frame, "ANCHOR", (c[0] + 20, c[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 3. Calibration
        M = get_homography(corners, ids)

        # Output Canvas (Start Black)
        projector_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        active_cards = []

        # --- STEP 1: LOCATE CARDS ---
        if ids is not None and M is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in CARD_LIBRARY:
                    card = CARD_LIBRARY[marker_id]

                    # Get center in Camera Space
                    cam_center = np.mean(corners[i][0], axis=0)

                    # Convert to Projector Space
                    cx, cy = transform_point(cam_center, M)
                    card.set_position(cx, cy)
                    active_cards.append(card)

        # --- STEP 2: COMPUTE LOGIC ---
        # Sort by X position so left-most cards (Sources) run first
        active_cards.sort(key=lambda c: c.center[0])
        for card in active_cards:
            try:
                card.compute(active_cards)
            except Exception as e:
                print(f"Math Error on card {card.id}: {e}")

        # --- STEP 3: DRAW PROJECTION ---
        for card in active_cards:
            cx, cy = card.center

            # A. Draw Connection Line (Visualizing the Flow)
            if card.input_source:
                sx, sy = card.input_source.center
                # Draw green line from Source to Me
                cv2.line(projector_canvas, (sx, sy), (cx, cy), (0, 255, 0), 2)

            # B. Draw The Image Result
            if card.output is not None:
                # Normalize float image to 0-255 uint8 for display
                disp_img = cv2.normalize(card.output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Resize to target display width (e.g. 250px wide)
                h, w = disp_img.shape
                aspect = h / w
                target_w = 250
                target_h = int(target_w * aspect)
                disp_img = cv2.resize(disp_img, (target_w, target_h))

                # Convert Grayscale to BGR
                disp_img_color = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)

                # Calculate coordinates to center image on card
                y1 = max(0, cy - target_h // 2)
                y2 = min(CANVAS_H, cy + target_h // 2)
                x1 = max(0, cx - target_w // 2)
                x2 = min(CANVAS_W, cx + target_w // 2)

                # Copy into canvas safely
                h_slice = y2 - y1
                w_slice = x2 - x1
                if h_slice > 0 and w_slice > 0:
                    projector_canvas[y1:y2, x1:x2] = disp_img_color[:h_slice, :w_slice]

            else:
                # C. Draw Waiting State (Red Outline if disconnected)
                if card.type != "source":
                    cv2.rectangle(projector_canvas, (cx - 60, cy - 60), (cx + 60, cy + 60), (0, 0, 255), 2)
                    cv2.putText(projector_canvas, "NO INPUT", (cx - 40, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- DISPLAY ---
        cv2.imshow("Realtalk Projector", projector_canvas)
        cv2.imshow("Camera Debug View", frame)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()