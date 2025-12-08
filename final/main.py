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

# Directions
LEFT = "LEFT"
RIGHT = "RIGHT"
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


# --- 2. CLASS LIBRARY (Refactored) ---

class BaseCard:
    """
    Base class with declarative inputs/outputs and dependency resolution logic.
    """

    def __init__(self, id, pos, width, height, inputs=None, outputs=None, is_virtual=False):
        self.id = id
        self.top_left = pos  # (x, y)
        self.width = width
        self.height = height
        self.is_virtual = is_virtual
        self.ttl = CARD_TTL

        # DECLARATIVE SPECS
        # list of tuples: [(Direction, ClassType)]
        self.inputs_spec = inputs or []
        # list of tuples: [(Direction, ClassType)]
        self.outputs_spec = outputs or []

        # RUNTIME STATE
        self.resolved_inputs = {}  # { Direction: CardObject }
        self.output_generated = None

        # Connection Visualization State
        self.conn_lines = []  # list of (start_pt, end_pt, is_connected) for drawing

    def is_point_inside(self, point):
        px, py = point
        x1, y1 = self.top_left
        x2 = x1 + self.width
        y2 = y1 + self.height
        return (x1 < px < x2) and (y1 < py < y2)

    def resolve_dependencies(self, active_cards):
        """
        Scans neighbors based on inputs_spec and populates resolved_inputs.
        """
        self.resolved_inputs = {}
        self.conn_lines = []

        x, y = self.top_left
        center_y = y + (self.height // 2)

        for direction, target_class in self.inputs_spec:
            probe_point = (0, 0)
            vis_start = (0, 0)

            # 1. Determine Probe Geometry (Fixed Length)
            if direction == LEFT:
                probe_point = (x - PROBE_DISTANCE, center_y)
                vis_start = (x, center_y)
            elif direction == RIGHT:
                probe_point = (x + self.width + PROBE_DISTANCE, center_y)
                vis_start = (x + self.width, center_y)

            # 2. Hit Test
            found_card = None
            # Visuals strictly follow geometry, no snapping to neighbor center
            vis_end = probe_point

            for candidate in active_cards:
                if candidate is not self and isinstance(candidate, target_class):
                    if candidate.is_point_inside(probe_point):
                        found_card = candidate
                        break  # Found a match, stop looking

            # 3. Store Result
            if found_card:
                self.resolved_inputs[direction] = found_card

            # 4. Store Visuals
            self.conn_lines.append((vis_start, vis_end, found_card is not None))

    def get_priority(self):
        """
        Calculates topological depth.
        0 = Leaf / No Dependencies met
        N = 1 + Max Depth of inputs
        """
        if not self.resolved_inputs:
            return 0

        # Priority is 1 + highest priority of any connected input
        max_input_prio = 0
        for direction, card in self.resolved_inputs.items():
            max_input_prio = max(max_input_prio, card.get_priority())

        return 1 + max_input_prio

    def project_output(self, direction, payload_object):
        """
        Helper to position the output virtual card.
        """
        x, y = self.top_left
        center_x = x + (self.width // 2)
        bottom_y = y + self.height

        if direction == DOWN:
            res_h = payload_object.height
            res_w = payload_object.width
            vx = center_x - (res_w // 2)
            vy = y + VIRTUAL_CARD_OFFSET_Y

            payload_object.top_left = (vx, vy)

            # Add visual line for output (Vertical)
            start = (center_x, bottom_y)
            end = (center_x, vy)  # Straight down to the top of the new card
            self.conn_lines.append((start, end, True))

            return payload_object
        return None

    def draw_connections(self, canvas):
        LINE_COLOR = (0, 255, 0)
        THICKNESS = 3
        RADIUS = 10

        for start, end, is_connected in self.conn_lines:
            cv2.line(canvas, start, end, LINE_COLOR, THICKNESS)
            # Draw circle at destination
            if is_connected:
                cv2.circle(canvas, end, RADIUS, LINE_COLOR, -1)  # Filled if connected
            else:
                cv2.circle(canvas, end, RADIUS, LINE_COLOR, THICKNESS)  # Hollow if searching
                cv2.circle(canvas, end, RADIUS - 3, (0, 0, 0), -1)

    def render_img(self, canvas, img):
        project_image_at(canvas, img, self.top_left)

    def reset_logic_state(self):
        self.output_generated = None
        self.resolved_inputs = {}
        self.conn_lines = []

    def run_logic(self, canvas, active_cards, id_generator=None):
        """
        Runs logic AND rendering.
        Returns a new Virtual Card if one is generated.
        """
        return None


class ImageCard(BaseCard):
    def __init__(self, id, pos, image_data, is_virtual=False):
        h, w = image_data.shape[:2] if image_data is not None else (0, 0)

        # ImageCard has NO inputs
        super().__init__(id, pos, width=w, height=h, inputs=[], outputs=[], is_virtual=is_virtual)

        self.img_arr = image_data
        self.type = "source"

    def run_logic(self, canvas, active_cards, id_generator=None):
        # Image Logic: Just Render
        if self.img_arr is not None:
            self.render_img(canvas, self.img_arr)
        return None


class KernelCard(BaseCard):
    def __init__(self, id, pos, kernel_arr, is_virtual=False):
        # KernelCard expects an ImageCard to the LEFT
        super().__init__(id, pos, width=200, height=200,
                         inputs=[(LEFT, ImageCard)],
                         outputs=[(DOWN, ImageCard)],
                         is_virtual=is_virtual)

        self.type = "kernel"
        self.kernel_arr = kernel_arr

    def run_logic(self, canvas, active_cards, id_generator=None):

        new_virtual_card = None

        # 1. EVALUATE LOGIC
        # Check if Dependency Resolved (populated by resolve_dependencies earlier)
        source = self.resolved_inputs.get(LEFT)

        if source and source.img_arr is not None:
            # Process (Convolution)
            result_arr = source.img_arr
            for _ in range(10):  # Multi-pass for visibility
                result_arr = cv2.filter2D(result_arr, -1, self.kernel_arr)

            # Determine ID
            vid = -1
            if id_generator:
                vid = id_generator()

            # Create Result Object
            result_card = ImageCard(vid, (0, 0), result_arr, is_virtual=True)

            # Position Output & Store
            new_virtual_card = self.project_output(DOWN, result_card)

        # 2. RENDER SELF
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
                    if abs(val - 1 / 9) < 0.001:
                        text = "1/9"
                    elif abs(val - 1.0) < 0.001:
                        text = "1"
                    elif abs(val) < 0.001:
                        text = "0"

                    center_x = int(c * step_x + step_x / 2)
                    center_y = int(r * step_y + step_y / 2)
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.putText(k_viz, text, (center_x - text_w // 2, center_y + text_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.rectangle(k_viz, (0, 0), (viz_w - 1, viz_h - 1), (0, 0, 0), 4)
        self.render_img(canvas, k_viz)

        return new_virtual_card


class KernelAdditionCard(BaseCard):
    """
    Takes two kernels (Left/Right) and outputs a summed Kernel.
    """

    def __init__(self, id, pos, is_virtual=False):
        super().__init__(id, pos, width=200, height=200,
                         inputs=[(LEFT, KernelCard), (RIGHT, KernelCard)],
                         outputs=[(DOWN, KernelCard)],
                         is_virtual=is_virtual)
        self.type = "operation"

    def run_logic(self, canvas, active_cards, id_generator=None):
        new_virtual_card = None

        # 1. EVALUATE LOGIC
        k_left = self.resolved_inputs.get(LEFT)
        k_right = self.resolved_inputs.get(RIGHT)

        if k_left and k_right:
            # Attempt addition
            try:
                # Simple addition (supports numpy broadcasting)
                sum_arr = k_left.kernel_arr + k_right.kernel_arr

                # Determine ID
                vid = -1
                if id_generator:
                    vid = id_generator()

                # Create Result (Virtual Kernel)
                # Note: We reuse KernelCard for the result type
                result_card = KernelCard(vid, (0, 0), kernel_arr=sum_arr, is_virtual=True)
                new_virtual_card = self.project_output(DOWN, result_card)
            except Exception:
                pass  # shape mismatch etc

        # 2. RENDER SELF
        # Draw a "Plus" visualization
        viz = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        cv2.rectangle(viz, (0, 0), (self.width - 1, self.height - 1), (0, 0, 0), 4)

        # Draw Plus Symbol
        center_x, center_y = self.width // 2, self.height // 2
        line_len = 40
        cv2.line(viz, (center_x - line_len, center_y), (center_x + line_len, center_y), (0, 0, 0), 5)
        cv2.line(viz, (center_x, center_y - line_len), (center_x, center_y + line_len), (0, 0, 0), 5)

        self.render_img(canvas, viz)

        return new_virtual_card


# --- 3. FACTORY & HELPERS ---

def create_physical_card(marker_id, pos):
    if marker_id == 10:
        return ImageCard(10, pos, SOURCE_IMG, is_virtual=False)
    elif marker_id == 20:
        k_size = 3
        blur_kernel = np.ones((k_size, k_size), np.float32) / (k_size ** 2)
        return KernelCard(20, pos, kernel_arr=blur_kernel, is_virtual=False)
    elif marker_id == 21:
        # Vertical Gradient (Sobel Y)
        vertical_grad_kernel = np.array([[-1, -2, -1],
                                         [0, 0, 0],
                                         [1, 2, 1]], dtype=np.float32)
        return KernelCard(21, pos, kernel_arr=vertical_grad_kernel, is_virtual=False)
    elif marker_id == 22:
        # Horizontal Gradient (Sobel X)
        horizonal_grad_kernel = np.array([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]], dtype=np.float32)
        return KernelCard(22, pos, kernel_arr=horizonal_grad_kernel, is_virtual=False)
    elif marker_id == 23:
        return KernelAdditionCard(23, pos, is_virtual=False)
    elif marker_id == 30:
        return KernelAdditionCard(30, pos, is_virtual=False)
    return None


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

        # --- UNIFIED LOGIC LOOP WITH TOPOLOGICAL SORT ---

        # ID Generator for Virtual Cards (resets every frame)
        virtual_id_counter = 0

        def get_next_virtual_id():
            nonlocal virtual_id_counter
            virtual_id_counter += 1
            return f"v-{virtual_id_counter}"

        # 1. Start with known physical cards
        active_cards = list(physical_cards_state.values())
        for c in active_cards: c.reset_logic_state()

        # Dummy canvas for logic passes (to prevent drawing trails/ghosts)
        dummy_canvas = np.zeros_like(projector_canvas)

        # 2. SIMULATION PHASE (Resolve dependencies & Generate Virtuals)
        for depth in range(MAX_CHAIN_DEPTH):

            # A. Resolve Dependencies
            for card in active_cards:
                card.resolve_dependencies(active_cards)

            # B. Sort
            active_cards.sort(key=lambda c: c.get_priority())

            # C. Run Logic (on dummy canvas)
            new_virtuals = []
            for card in active_cards:
                if card.output_generated is None:
                    # Pass the ID generator to the logic function
                    out = card.run_logic(dummy_canvas, active_cards, id_generator=get_next_virtual_id)
                    if out:
                        new_virtuals.append(out)
                        card.output_generated = out
                else:
                    # Update logic state for existing chains if needed
                    card.run_logic(dummy_canvas, active_cards, id_generator=get_next_virtual_id)

            if not new_virtuals:
                break

            active_cards.extend(new_virtuals)

        # 3. RENDER PHASE (Final Draw)
        # Now that the graph is stable, we do ONE pass to draw everything cleanly.
        for card in active_cards:
            # Re-resolve dependencies to ensure connection lines are fresh/correct for this frame
            card.resolve_dependencies(active_cards)

            # Run logic one last time on the REAL canvas to draw content & add output lines
            # (We pass the generator, but new IDs shouldn't typically be used here as outputs are already generated)
            card.run_logic(projector_canvas, active_cards, id_generator=get_next_virtual_id)

            # Draw the connection lines
            card.draw_connections(projector_canvas)

        # --- DEBUG OVERLAY ---
        if DEBUG_MODE and M_inv is not None:
            for card in active_cards:
                # Draw input probe visualization (if it exists)
                for start_pt, end_pt, is_connected in card.conn_lines:
                    cam_tip = transform_point(end_pt, M_inv)
                    cv2.circle(frame, cam_tip, 6, (0, 165, 255), -1)

                if card.width > 0:
                    w, h = card.width, card.height
                    x1, y1 = card.top_left
                    x2, y2 = x1 + w, y1 + h
                    proj_pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype='float32')
                    cam_pts = cv2.perspectiveTransform(proj_pts, M_inv).astype(int)

                    color = (0, 255, 255) if card.is_virtual else (255, 0, 0)
                    cv2.polylines(frame, [cam_pts], True, color, 2)

                    # Update Label to show ID
                    label = f"VIRTUAL ({card.id})" if card.is_virtual else f"PHYSICAL ({card.id})"
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