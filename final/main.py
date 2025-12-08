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


# --- 2. CARD DATABASE (Server) ---

class CardSystem:
    """
    Acts as the database/server managing the state (ID, Position, TTL)
    for all card instances. Decouples state from behavior.
    """
    _store = {}  # instance -> {id, pos, is_virtual, ttl}
    _virtual_id_counter = 0
    canvas = None  # Current frame canvas context

    @classmethod
    def register(cls, card, id=None, pos=(0, 0), is_virtual=False):
        # Auto-generate ID for virtuals if not provided
        if id is None and is_virtual:
            cls._virtual_id_counter += 1
            id = f"v-{cls._virtual_id_counter}"

        cls._store[card] = {
            'id': id,
            'pos': pos,
            'is_virtual': is_virtual,
            'ttl': CARD_TTL
        }
        return id

    @classmethod
    def update_physical(cls, card, pos):
        if card in cls._store:
            entry = cls._store[card]

            # Smoothing Logic: Low-pass filter to reduce jitter
            old_x, old_y = entry['pos']
            target_x, target_y = pos

            smooth_x = int(old_x * 0.8 + target_x * 0.2)
            smooth_y = int(old_y * 0.8 + target_y * 0.2)

            entry['pos'] = (smooth_x, smooth_y)
            entry['ttl'] = CARD_TTL  # Reset TTL on active tracking

    @classmethod
    def get_pos(cls, card):
        return cls._store[card]['pos'] if card in cls._store else (0, 0)

    @classmethod
    def get_id(cls, card):
        return cls._store[card]['id'] if card in cls._store else "?"

    @classmethod
    def is_virtual(cls, card):
        return cls._store[card]['is_virtual'] if card in cls._store else False

    @classmethod
    def get_ttl(cls, card):
        return cls._store[card]['ttl'] if card in cls._store else 0

    @classmethod
    def decrease_ttl(cls, card):
        if card in cls._store:
            cls._store[card]['ttl'] -= 1

    @classmethod
    def unregister(cls, card):
        if card in cls._store:
            del cls._store[card]

    @classmethod
    def reset_frame(cls, canvas):
        cls.canvas = canvas
        cls._virtual_id_counter = 0
        # Optional: We could garbage collect old virtuals here if we tracked them explicitly


# --- 3. CLASS LIBRARY (Refactored) ---

class BaseCard:
    """
    Base class with declarative inputs/outputs and dependency resolution logic.
    State is delegated to CardSystem.
    """

    def __init__(self, width, height, inputs=None, outputs=None):
        self.width = width
        self.height = height

        # DECLARATIVE SPECS
        self.inputs_spec = inputs or []
        self.outputs_spec = outputs or []

        # RUNTIME STATE (Logic only)
        self.resolved_inputs = {}  # { Direction: CardObject }
        self.output_generated = None
        self.conn_lines = []

        # --- Properties delegating to Database ---

    @property
    def top_left(self):
        return CardSystem.get_pos(self)

    @property
    def id(self):
        return CardSystem.get_id(self)

    @property
    def is_virtual(self):
        return CardSystem.is_virtual(self)

    @property
    def ttl(self):
        return CardSystem.get_ttl(self)

    # --- Geometry & Resolution ---

    def is_point_inside(self, point):
        px, py = point
        x1, y1 = self.top_left
        x2 = x1 + self.width
        y2 = y1 + self.height
        return (x1 < px < x2) and (y1 < py < y2)

    def resolve_dependencies(self, active_cards):
        self.resolved_inputs = {}
        self.conn_lines = []

        x, y = self.top_left
        center_y = y + (self.height // 2)

        for direction, target_class in self.inputs_spec:
            probe_point = (0, 0)
            vis_start = (0, 0)

            # 1. Determine Probe Geometry
            if direction == LEFT:
                probe_point = (x - PROBE_DISTANCE, center_y)
                vis_start = (x, center_y)
            elif direction == RIGHT:
                probe_point = (x + self.width + PROBE_DISTANCE, center_y)
                vis_start = (x + self.width, center_y)

            # 2. Hit Test
            found_card = None
            vis_end = probe_point

            for candidate in active_cards:
                if candidate is not self and isinstance(candidate, target_class):
                    if candidate.is_point_inside(probe_point):
                        found_card = candidate
                        break

                        # 3. Store Result
            if found_card:
                self.resolved_inputs[direction] = found_card

            # 4. Store Visuals
            self.conn_lines.append((vis_start, vis_end, found_card is not None))

    def get_priority(self):
        if not self.resolved_inputs:
            return 0

        max_input_prio = 0
        for direction, card in self.resolved_inputs.items():
            max_input_prio = max(max_input_prio, card.get_priority())

        return 1 + max_input_prio

    def project_output(self, direction, payload_card):
        """
        Calculates position for the new card, registers it in DB,
        and adds the output connection line.
        """
        x, y = self.top_left
        center_x = x + (self.width // 2)
        bottom_y = y + self.height

        if direction == DOWN:
            # Calculate geometric position
            vx = center_x - (payload_card.width // 2)
            vy = y + VIRTUAL_CARD_OFFSET_Y

            target_card = payload_card

            # Handle Frame Consistency: reuse existing output if available
            # This allows run_logic to be "stateless" (create new objects)
            # while the system maintains continuity.
            if self.output_generated is not None:
                target_card = self.output_generated
                # Update content if applicable
                if hasattr(target_card, 'img_arr') and hasattr(payload_card, 'img_arr'):
                    target_card.img_arr = payload_card.img_arr
                elif hasattr(target_card, 'kernel_arr') and hasattr(payload_card, 'kernel_arr'):
                    target_card.kernel_arr = payload_card.kernel_arr
            else:
                self.output_generated = target_card
                # Register with the System (Assigns ID, Stores Pos)
                CardSystem.register(target_card, pos=(vx, vy), is_virtual=True)

            # Always Ensure position is up to date in System
            if target_card in CardSystem._store:
                CardSystem._store[target_card]['pos'] = (vx, vy)

            # Add visual line for output
            start = (center_x, bottom_y)
            end = (center_x, vy)
            self.conn_lines.append((start, end, True))

            return target_card
        return None

    def draw_connections(self):
        if CardSystem.canvas is None: return

        LINE_COLOR = (0, 255, 0)
        THICKNESS = 3
        RADIUS = 10

        for start, end, is_connected in self.conn_lines:
            cv2.line(CardSystem.canvas, start, end, LINE_COLOR, THICKNESS)
            if is_connected:
                cv2.circle(CardSystem.canvas, end, RADIUS, LINE_COLOR, -1)
            else:
                cv2.circle(CardSystem.canvas, end, RADIUS, LINE_COLOR, THICKNESS)
                cv2.circle(CardSystem.canvas, end, RADIUS - 3, (0, 0, 0), -1)

    def render_img(self, img):
        if CardSystem.canvas is None: return
        project_image_at(CardSystem.canvas, img, self.top_left)

    def reset_logic_state(self):
        self.output_generated = None
        self.resolved_inputs = {}
        self.conn_lines = []

    def run_logic(self):
        """
        User Script. Pure behavior.
        """
        pass


class ImageCard(BaseCard):
    def __init__(self, image_data):
        h, w = image_data.shape[:2] if image_data is not None else (0, 0)
        super().__init__(width=w, height=h, inputs=[], outputs=[])
        self.img_arr = image_data
        self.type = "source"

    def run_logic(self):
        # Logic: None
        # Render: Self
        if self.img_arr is not None:
            self.render_img(self.img_arr)


class KernelCard(BaseCard):
    def __init__(self, kernel_arr):
        super().__init__(width=200, height=200,
                         inputs=[(LEFT, ImageCard)],
                         outputs=[(DOWN, ImageCard)])
        self.type = "kernel"
        self.kernel_arr = kernel_arr

    def run_logic(self):
        # 1. RENDER SELF (Visual Matrix)
        self._render_matrix_visual()

        # 2. LOGIC
        source = self.resolved_inputs.get(LEFT)
        if not source or source.img_arr is None:
            return

        # Process
        result_arr = source.img_arr
        for _ in range(10):
            result_arr = cv2.filter2D(result_arr, -1, self.kernel_arr)

        # Result Creation (Clean - No IDs/Pos)
        result_card = ImageCard(result_arr)

        self.project_output(DOWN, result_card)

    def _render_matrix_visual(self):
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
        self.render_img(k_viz)


class KernelAdditionCard(BaseCard):
    def __init__(self):
        super().__init__(width=200, height=200,
                         inputs=[(LEFT, KernelCard), (RIGHT, KernelCard)],
                         outputs=[(DOWN, KernelCard)])
        self.type = "operation"

    def run_logic(self):
        # 1. RENDER SELF
        self._render_plus_visual()

        # 2. LOGIC
        k_left = self.resolved_inputs.get(LEFT)
        k_right = self.resolved_inputs.get(RIGHT)

        if not (k_left and k_right):
            return

        try:
            sum_arr = k_left.kernel_arr + k_right.kernel_arr
            # Clean Creation
            result_card = KernelCard(sum_arr)
            self.project_output(DOWN, result_card)
        except Exception:
            return

    def _render_plus_visual(self):
        viz = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        cv2.rectangle(viz, (0, 0), (self.width - 1, self.height - 1), (0, 0, 0), 4)
        center_x, center_y = self.width // 2, self.height // 2
        line_len = 40
        cv2.line(viz, (center_x - line_len, center_y), (center_x + line_len, center_y), (0, 0, 0), 5)
        cv2.line(viz, (center_x, center_y - line_len), (center_x, center_y + line_len), (0, 0, 0), 5)
        self.render_img(viz)


# --- 4. FACTORY & HELPERS ---

def create_physical_card_instance(marker_id):
    """
    Creates the logic instance. Registration happens in Main.
    """
    if marker_id == 10:
        return ImageCard(SOURCE_IMG)
    elif marker_id == 20:
        k_size = 3
        blur_kernel = np.ones((k_size, k_size), np.float32) / (k_size ** 2)
        return KernelCard(blur_kernel)
    elif marker_id == 21:
        # Vertical Gradient (Sobel Y)
        vertical_grad_kernel = np.array([[-1, -2, -1],
                                         [0, 0, 0],
                                         [1, 2, 1]], dtype=np.float32)
        return KernelCard(vertical_grad_kernel)
    elif marker_id == 22:
        # Horizontal Gradient (Sobel X)
        horizonal_grad_kernel = np.array([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]], dtype=np.float32)
        return KernelCard(horizonal_grad_kernel)
    elif marker_id == 23 or marker_id == 30:
        return KernelAdditionCard()
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

    # Persistent Map: Marker ID -> Card Instance
    physical_cards_map = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        corners, ids, rejected = detector.detectMarkers(frame)
        M = get_homography(corners, ids)
        M_inv = np.linalg.inv(M) if M is not None else None

        projector_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # 1. SETUP CONTEXT
        CardSystem.reset_frame(projector_canvas)

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

                    # Existence Check
                    if mid not in physical_cards_map:
                        new_card = create_physical_card_instance(mid)
                        if new_card:
                            physical_cards_map[mid] = new_card
                            # Register with DB
                            CardSystem.register(new_card, id=str(mid), pos=(cx, cy), is_virtual=False)

                    # Update Position in DB
                    card = physical_cards_map.get(mid)
                    if card:
                        # Smooth position logic (low-pass filter)
                        CardSystem.update_physical(card, (cx, cy))

        # Decay Logic
        keys_to_remove = []
        for mid, card in physical_cards_map.items():
            if mid not in seen_marker_ids:
                CardSystem.decrease_ttl(card)
                if CardSystem.get_ttl(card) <= 0:
                    keys_to_remove.append(mid)

        for k in keys_to_remove:
            card = physical_cards_map[k]
            CardSystem.unregister(card)
            del physical_cards_map[k]

        # --- UNIFIED LOGIC LOOP ---

        active_cards = list(physical_cards_map.values())
        for c in active_cards: c.reset_logic_state()

        # Dummy canvas for logic passes
        dummy_canvas = np.zeros_like(projector_canvas)

        # 2. SIMULATION PHASE
        for depth in range(MAX_CHAIN_DEPTH):
            # A. Resolve
            for card in active_cards:
                card.resolve_dependencies(active_cards)

            # B. Sort
            active_cards.sort(key=lambda c: c.get_priority())

            # C. Run Logic (No Render, only state update)
            # Temporarily swap context to dummy to prevent drawing during calculation
            CardSystem.canvas = dummy_canvas

            new_virtuals = []
            for card in active_cards:
                # We check the internal state to see if output was created
                if card.output_generated is None:
                    card.run_logic()
                    if card.output_generated:
                        new_virtuals.append(card.output_generated)
                else:
                    card.run_logic()

            if not new_virtuals:
                break
            active_cards.extend(new_virtuals)

        # 3. RENDER PHASE (Final Draw)
        CardSystem.canvas = projector_canvas  # Restore real canvas

        for card in active_cards:
            # Re-resolve for fresh lines
            card.resolve_dependencies(active_cards)

            # Run logic to Draw Content & Lines
            card.run_logic()
            card.draw_connections()

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

                    label = f"{card.id}"
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