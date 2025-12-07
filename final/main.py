import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
CANVAS_W, CANVAS_H = 1920, 1080
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
ANCHOR_IDS = [0, 1, 2, 3]
DEBUG_MODE = True
WINDOW_NAME = "Projector Output"


# --- 1. REPOSITORY LAYER (Simulated) ---

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


# Mock Database
SOURCE_IMG = load_image_resource("A.jpg", width=250)
KERNEL_IMG = load_image_resource("blur3.png", width=200)


# --- 2. KERNEL LAYER (The "Card" Class) ---

class Card:
    def __init__(self, id, pos, type, image_data=None):
        self.id = id
        self.top_left = pos  # tuple (x, y) - Defines the Top-Left of the Valid Area
        self.type = type  # "source" or "kernel"
        self.data = image_data

        if image_data is not None:
            self.height, self.width, _ = image_data.shape
        else:
            self.width, self.height = 0, 0

        # Logic State
        self.input_neighbor = None
        self.output_result = None

        # Visual State
        self.connector_tip = None


def create_card_from_id(marker_id, pos):
    if marker_id == 10:
        return Card(10, pos, "source", SOURCE_IMG)
    elif marker_id == 20:
        return Card(20, pos, "kernel", KERNEL_IMG)
    return None


def is_point_in_card(point, card):
    """
    Hit-test helper: Checks if point is within the card's Valid Area.
    Valid Area = Rectangle starting at card.top_left with card.width/height.
    """
    px, py = point
    x1, y1 = card.top_left
    x2 = x1 + card.width
    y2 = y1 + card.height

    return (x1 < px < x2) and (y1 < py < y2)


def solve_relationships(active_cards):
    """
    Calculates connections based on physical overlap with the Probe.
    """
    PROBE_DISTANCE = 250  # Pixels to look to the left

    # Define a 15x15 Box Blur Kernel
    k_size = 15
    blur_kernel = np.ones((k_size, k_size), np.float32) / (k_size ** 2)

    for card in active_cards:
        if card.type == "kernel":
            # A. Calculate "Ideal" Probe
            # Logic: Start at Kernel's Left Edge, Center Y
            kernel_left_x = card.top_left[0]
            kernel_center_y = card.top_left[1] + (card.height // 2)

            probe_x = kernel_left_x - PROBE_DISTANCE
            probe_y = kernel_center_y

            best_neighbor = None
            snap_point = (probe_x, probe_y)  # Default: No snap

            # B. Hit Test: Is the probe inside the Valid Area of a source?
            for candidate in active_cards:
                if candidate.type == "source":
                    if is_point_in_card((probe_x, probe_y), candidate):
                        best_neighbor = candidate

                        # C. Snap Point: Candidate's Right Edge, Center Y
                        src_right_x = candidate.top_left[0] + candidate.width
                        src_center_y = candidate.top_left[1] + (candidate.height // 2)
                        snap_point = (src_right_x, src_center_y)
                        break

                        # D. Update State
            card.input_neighbor = best_neighbor
            card.connector_tip = snap_point

            if best_neighbor and best_neighbor.data is not None:
                convolved_img = cv2.filter2D(best_neighbor.data, -1, blur_kernel)
                card.output_result = convolved_img
            else:
                card.output_result = None


# --- 3. VISION LAYER (The Renderer) ---

def project_image_at(canvas, img, top_left):
    """
    Draws image extending Down and Right from the top_left coordinate.
    """
    h, w, _ = img.shape
    x1, y1 = top_left
    x2 = x1 + w
    y2 = y1 + h

    # Canvas bounds
    c_y1, c_y2 = max(0, y1), min(CANVAS_H, y2)
    c_x1, c_x2 = max(0, x1), min(CANVAS_W, x2)

    # Image bounds (offset logic)
    i_y1, i_y2 = c_y1 - y1, h - (y2 - c_y2)
    i_x1, i_x2 = c_x1 - x1, w - (x2 - c_x2)

    if c_x2 > c_x1 and c_y2 > c_y1:
        canvas[c_y1:c_y2, c_x1:c_x2] = img[i_y1:i_y2, i_x1:i_x2]

    return w, h


def draw_kernel_overlay(canvas, card):
    """
    Draws lines based on the calculated 'connector_tip' and card geometry.
    """
    x, y = card.top_left
    w, h = card.width, card.height

    # Calculate geometric points relative to Top-Left
    kernel_left_x = x
    kernel_center_y = y + (h // 2)
    kernel_bottom_y = y + h
    kernel_center_x = x + (w // 2)

    LINE_COLOR = (0, 255, 0)
    THICKNESS = 3
    RADIUS = 10

    # --- 1. INPUT VISUALIZATION (Left) ---
    if card.connector_tip:
        tip_x, tip_y = card.connector_tip

        # Draw Line: Kernel Left Edge -> Tip
        cv2.line(canvas, (kernel_left_x, kernel_center_y), (tip_x, tip_y), LINE_COLOR, THICKNESS)

        # Draw Hollow Circle at the Tip
        cv2.circle(canvas, (tip_x, tip_y), RADIUS, LINE_COLOR, THICKNESS)
        cv2.circle(canvas, (tip_x, tip_y), RADIUS - 3, (0, 0, 0), -1)

        if not card.input_neighbor:
            cv2.putText(canvas, "Place Image Here", (tip_x - 60, tip_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # --- 2. OUTPUT VISUALIZATION (Down) ---
    if card.output_result is not None:
        start_x = kernel_center_x
        start_y = kernel_bottom_y

        OFFSET_DOWN = 220
        end_y = start_y + OFFSET_DOWN

        # Line down from bottom edge
        cv2.line(canvas, (start_x, start_y), (start_x, end_y), LINE_COLOR, THICKNESS)
        cv2.circle(canvas, (start_x, end_y), RADIUS, LINE_COLOR, -1)

        # Draw Result
        res_h, res_w, _ = card.output_result.shape
        # Center the result image horizontally on the line end, but start drawing vertically below it
        res_top_left = (start_x - res_w // 2, end_y + 20)
        project_image_at(canvas, card.output_result, res_top_left)


# --- 4. CORE ENGINE HELPERS ---

def get_homography(corners, ids):
    if ids is None: return None
    found_anchors = {}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in ANCHOR_IDS:
            found_anchors[marker_id] = np.mean(corners[i][0], axis=0)
    if len(found_anchors) < 4: return None
    src = np.array([found_anchors[0], found_anchors[1], found_anchors[2], found_anchors[3]], dtype="float32")
    dst = np.array([[0, 0], [CANVAS_W, 0], [CANVAS_W, CANVAS_H], [0, CANVAS_H]], dtype="float32")
    return cv2.getPerspectiveTransform(src, dst)


def transform_point(point, M):
    pt = np.array([[[point[0], point[1]]]], dtype='float32')
    t = cv2.perspectiveTransform(pt, M)
    return int(t[0][0][0]), int(t[0][0][1])


# --- 5. MAIN LOOP ---

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("System Started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        corners, ids, rejected = detector.detectMarkers(frame)
        M = get_homography(corners, ids)
        M_inv = np.linalg.inv(M) if M is not None else None

        # Reset Canvas
        projector_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        if M is not None:
            c = (50, 50, 50)
            cv2.line(projector_canvas, (0, 0), (50, 0), c, 2);
            cv2.line(projector_canvas, (0, 0), (0, 50), c, 2)
            cv2.line(projector_canvas, (CANVAS_W, 0), (CANVAS_W - 50, 0), c, 2);
            cv2.line(projector_canvas, (CANVAS_W, 0), (CANVAS_W, 50), c, 2)
            cv2.line(projector_canvas, (0, CANVAS_H), (50, CANVAS_H), c, 2);
            cv2.line(projector_canvas, (0, CANVAS_H), (0, CANVAS_H - 50), c, 2)
            cv2.line(projector_canvas, (CANVAS_W, CANVAS_H), (CANVAS_W - 50, CANVAS_H), c, 2);
            cv2.line(projector_canvas, (CANVAS_W, CANVAS_H), (CANVAS_W, CANVAS_H - 50), c, 2)

        active_cards = []

        if ids is not None and M is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in ANCHOR_IDS:
                    marker_corners = corners[i][0]
                    marker_btm_left = marker_corners[3]

                    cx, cy = transform_point(marker_btm_left, M)

                    # Create card using this top-left position
                    new_card = create_card_from_id(int(marker_id), (cx, cy))
                    if new_card:
                        active_cards.append(new_card)

        solve_relationships(active_cards)

        for card in active_cards:
            # Draw Card Image (Top-Left based)
            if card.data is not None:
                project_image_at(projector_canvas, card.data, card.top_left)

            if card.type == "kernel":
                draw_kernel_overlay(projector_canvas, card)

            if DEBUG_MODE and M_inv is not None and card.width > 0:
                # Calculate Valid Area Box in Camera Space
                w, h = card.width, card.height
                x1, y1 = card.top_left
                x2, y2 = x1 + w, y1 + h

                # Order: TL, TR, BR, BL
                proj_pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype='float32')
                cam_pts = cv2.perspectiveTransform(proj_pts, M_inv).astype(int)
                cv2.polylines(frame, [cam_pts], True, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, projector_canvas)
        if DEBUG_MODE:
            cv2.imshow("Debug Input", frame)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()