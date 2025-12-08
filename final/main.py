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
PROBE_DISTANCE = 250
MAX_CHAIN_DEPTH = 5
CARD_TTL = 15

# Directions
LEFT = "LEFT"
DOWN = "DOWN"


# --- 1. REPOSITORY LAYER ---

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


SOURCE_IMG = load_image_resource("A.jpg", width=250)
KERNEL_IMG = load_image_resource("blur3.png", width=200)


# --- 2. CLASS LIBRARY (Refactored) ---

class BaseCard:
    """
    Base class defining the valid area, connectivity logic, and drawing.
    """

    def __init__(self, id, pos, width, height, is_virtual=False):
        self.id = id
        self.top_left = pos  # (x, y)
        self.width = width
        self.height = height
        self.is_virtual = is_virtual
        self.ttl = CARD_TTL

        # Connection Visualization State
        self.conn_input_start = None
        self.conn_input_end = None
        self.conn_output_start = None
        self.conn_output_end = None

        # Logic State
        self.output_generated = None

    def is_point_inside(self, point):
        px, py = point
        x1, y1 = self.top_left
        x2 = x1 + self.width
        y2 = y1 + self.height
        return (x1 < px < x2) and (y1 < py < y2)

    def input(self, direction, target_type_class, active_cards):
        """
        Looks for a neighbor card in the given direction.
        """
        x, y = self.top_left
        center_y = y + (self.height // 2)

        probe_point = (0, 0)

        # 1. Determine Probe Point
        if direction == LEFT:
            probe_point = (x - PROBE_DISTANCE, center_y)
            self.conn_input_start = (x, center_y)  # Draw from left edge
            self.conn_input_end = probe_point  # To probe point
        # (Add other directions here if needed later)

        # 2. Hit Test against Active Cards
        found_card = None
        for candidate in active_cards:
            if candidate is not self and isinstance(candidate, target_type_class):
                if candidate.is_point_inside(probe_point):
                    found_card = candidate

                    # Snap visual connection to candidate's edge
                    if direction == LEFT:
                        src_right_x = candidate.top_left[0] + candidate.width
                        src_center_y = candidate.top_left[1] + (candidate.height // 2)
                        self.conn_input_end = (src_right_x, src_center_y)
                    break

        return found_card

    def output(self, direction, payload_object):
        """
        Outputs a class object (Virtual Card) in the given direction.
        """
        x, y = self.top_left
        center_x = x + (self.width // 2)
        bottom_y = y + self.height

        new_pos = (0, 0)

        # 1. Determine Output Position
        if direction == DOWN:
            # Center the result horizontally below
            res_h = payload_object.height
            res_w = payload_object.width

            vx = center_x - (res_w // 2)
            vy = y + VIRTUAL_CARD_OFFSET_Y
            new_pos = (vx, vy)

            # Set visuals
            self.conn_output_start = (center_x, bottom_y)
            self.conn_output_end = (center_x + (vx - center_x) + (res_w // 2), vy)

        # 2. Update the Payload Object's position
        payload_object.top_left = new_pos

        return payload_object

    def draw_connections(self, canvas):
        LINE_COLOR = (0, 255, 0)
        THICKNESS = 3
        RADIUS = 10

        # Draw Input Line
        if self.conn_input_start and self.conn_input_end:
            cv2.line(canvas, self.conn_input_start, self.conn_input_end, LINE_COLOR, THICKNESS)
            cv2.circle(canvas, self.conn_input_end, RADIUS, LINE_COLOR, THICKNESS)
            cv2.circle(canvas, self.conn_input_end, RADIUS - 3, (0, 0, 0), -1)

        # Draw Output Line
        if self.conn_output_start and self.conn_output_end:
            cv2.line(canvas, self.conn_output_start, self.conn_output_end, LINE_COLOR, THICKNESS)
            cv2.circle(canvas, self.conn_output_end, RADIUS, LINE_COLOR, -1)

    def render(self, canvas):
        """
        Draws the card's visual content. Overridden by subclasses.
        """
        pass

    def reset_logic_state(self):
        self.output_generated = None
        self.conn_input_start = None
        self.conn_input_end = None
        self.conn_output_start = None
        self.conn_output_end = None


class ImageCard(BaseCard):
    """
    Represents data (an image). Can be physical or virtual.
    """

    def __init__(self, id, pos, image_data, is_virtual=False):
        h, w = image_data.shape[:2] if image_data is not None else (0, 0)
        super().__init__(id, pos, width=w, height=h, is_virtual=is_virtual)
        self.img_arr = image_data
        self.type = "source"  # Keep type tag for debug compatibility

    def render(self, canvas):
        # Draw the image data
        project_image_at(canvas, self.img_arr, self.top_left)


class KernelCard(BaseCard):
    """
    Represents an operation.
    """

    def __init__(self, id, pos, image_data, is_virtual=False):
        h, w = image_data.shape[:2] if image_data is not None else (0, 0)
        super().__init__(id, pos, width=w, height=h, is_virtual=is_virtual)
        self.kernel_arr = None  # Placeholder for actual matrix if needed
        self.img_representation = image_data  # Visual icon
        self.type = "kernel"

    def convolution(self, source_card):
        if source_card.img_arr is None: return None

        # Box blur logic
        k_size = 15
        blur_kernel = np.ones((k_size, k_size), np.float32) / (k_size ** 2)
        result_arr = cv2.filter2D(source_card.img_arr, -1, blur_kernel)

        # Return a new Virtual ImageCard (ID -1)
        return ImageCard(-1, (0, 0), result_arr, is_virtual=True)

    def run_logic(self, active_cards):
        """
        The main execution block for this card.
        """
        # 1. INPUT: Look LEFT for an ImageCard
        source = self.input(LEFT, ImageCard, active_cards)

        if source:
            # 2. PROCESS: Convolve
            result_card = self.convolution(source)

            # 3. OUTPUT: Send result DOWN
            if result_card:
                final_card = self.output(DOWN, result_card)
                self.output_generated = final_card
                return final_card

        return None

    def render(self, canvas):
        # Draw the kernel icon/representation
        project_image_at(canvas, self.img_representation, self.top_left)


# --- 3. FACTORY & HELPERS ---

def create_physical_card(marker_id, pos):
    if marker_id == 10:
        return ImageCard(10, pos, SOURCE_IMG, is_virtual=False)
    elif marker_id == 20:
        return KernelCard(20, pos, KERNEL_IMG, is_virtual=False)
    return None


def resolve_interactions(all_active_cards):
    """
    Iterative solver calling .run_logic() on KernelCards.
    """
    for depth in range(MAX_CHAIN_DEPTH):
        new_virtuals_this_pass = []

        for card in all_active_cards:
            # Run logic only on Kernels that haven't fired yet
            if isinstance(card, KernelCard) and card.output_generated is None:
                new_virtual = card.run_logic(all_active_cards)
                if new_virtual:
                    new_virtuals_this_pass.append(new_virtual)

        if not new_virtuals_this_pass:
            break

        all_active_cards.extend(new_virtuals_this_pass)


# --- 4. VISION & RENDER LAYER ---

def project_image_at(canvas, img, top_left):
    """Draws image safely within canvas bounds"""
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

    # Persistent state for physical cards
    physical_cards_state = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        corners, ids, rejected = detector.detectMarkers(frame)
        M = get_homography(corners, ids)
        M_inv = np.linalg.inv(M) if M is not None else None

        projector_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # --- DEBUG VISUALIZATION ---
        if DEBUG_MODE and corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in ANCHOR_IDS:
                    pt = np.mean(corners[i][0], axis=0).astype(int)
                    cv2.circle(frame, tuple(pt), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "Anchor", (pt[0] + 10, pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- UPDATE PHYSICAL CARDS ---
        seen_marker_ids = set()
        if ids is not None and M is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in ANCHOR_IDS:
                    mid = int(marker_id)
                    seen_marker_ids.add(mid)

                    marker_corners = corners[i][0]
                    marker_btm_left = marker_corners[3]
                    cx, cy = transform_point(marker_btm_left, M)

                    if mid in physical_cards_state:
                        # Update existing
                        card = physical_cards_state[mid]
                        # Smooth position
                        old_x, old_y = card.top_left
                        smooth_x = int(old_x * 0.8 + cx * 0.2)
                        smooth_y = int(old_y * 0.8 + cy * 0.2)
                        card.top_left = (smooth_x, smooth_y)
                        card.ttl = CARD_TTL
                    else:
                        # Create new
                        new_card = create_physical_card(mid, (cx, cy))
                        if new_card:
                            physical_cards_state[mid] = new_card

        # Decay
        keys_to_remove = []
        for mid, card in physical_cards_state.items():
            if mid not in seen_marker_ids:
                card.ttl -= 1
                if card.ttl <= 0:
                    keys_to_remove.append(mid)
        for k in keys_to_remove:
            del physical_cards_state[k]

        # --- RUN LOGIC ---
        active_cards = list(physical_cards_state.values())

        # Reset transient state
        for card in active_cards:
            card.reset_logic_state()

        # Resolve
        resolve_interactions(active_cards)

        # --- RENDER ---
        for card in active_cards:
            # 1. Draw Content (Polymorphic)
            card.render(projector_canvas)

            # 2. Draw Connections
            card.draw_connections(projector_canvas)

            # 3. Draw Debug Hitboxes
            if DEBUG_MODE and M_inv is not None:
                # Probe point logic is now inside card.conn_input_end
                if card.conn_input_end:
                    cam_tip = transform_point(card.conn_input_end, M_inv)
                    cv2.circle(frame, cam_tip, 6, (0, 165, 255), -1)

                if card.width > 0:
                    w, h = card.width, card.height
                    x1, y1 = card.top_left
                    x2, y2 = x1 + w, y1 + h
                    proj_pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype='float32')
                    cam_pts = cv2.perspectiveTransform(proj_pts, M_inv).astype(int)

                    color = (0, 255, 255) if card.is_virtual else (255, 0, 0)
                    cv2.polylines(frame, [cam_pts], True, color, 2)
                    label = "VIRTUAL" if card.is_virtual else "PHYSICAL"
                    cv2.putText(frame, label, tuple(cam_pts[0][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow(WINDOW_NAME, projector_canvas)
        if DEBUG_MODE:
            cv2.imshow("Debug Input", frame)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()