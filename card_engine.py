import cv2
import numpy as np
import types
import json
from PIL import Image
import common

# --- CARD DATABASE (Server) ---

class CardSystem:
    _store = {}
    _virtual_id_counter = 0
    canvas = None

    @classmethod
    def register(cls, card, id=None, pos=(0, 0), is_virtual=False):
        if id is None and is_virtual:
            cls._virtual_id_counter += 1
            id = f"v-{cls._virtual_id_counter}"

        cls._store[card] = {
            'id': id,
            'pos': pos,
            'is_virtual': is_virtual,
            'ttl': common.CARD_TTL
        }
        return id

    @classmethod
    def update_physical(cls, card, pos):
        if card in cls._store:
            entry = cls._store[card]
            old_x, old_y = entry['pos']
            target_x, target_y = pos
            smooth_x = int(old_x * 0.8 + target_x * 0.2)
            smooth_y = int(old_y * 0.8 + target_y * 0.2)
            entry['pos'] = (smooth_x, smooth_y)
            entry['ttl'] = common.CARD_TTL

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

# --- CLASS LIBRARY ---

class BaseCard:
    def __init__(self, width, height, inputs=None, outputs=None):
        self.width = width
        self.height = height
        self.inputs_spec = inputs or []
        self.outputs_spec = outputs or []
        self.resolved_inputs = {}
        self.input_ttls = {}
        self.output_generated = None
        self.conn_lines = []
        self.load_universal_script()

    def load_universal_script(self):
        script = CLASS_SCRIPTS.get(type(self))
        if script:
            self.load_script(script)

    def load_script(self, code_str):
        if not code_str: return
        indented_code = "\n".join(["    " + line for line in code_str.split("\n")])
        wrapped_code = f"def run_logic(self):\n{indented_code}"
        local_scope = {}
        exec(wrapped_code, globals(), local_scope)
        if "run_logic" in local_scope:
            setattr(self, "run_logic", types.MethodType(local_scope["run_logic"], self))

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

    def is_point_inside(self, point):
        px, py = point
        x1, y1 = self.top_left
        x2 = x1 + self.width
        y2 = y1 + self.height
        return (x1 < px < x2) and (y1 < py < y2)

    def resolve_dependencies(self, active_cards):
        self.conn_lines = []
        x, y = self.top_left
        center_y = y + (self.height // 2)

        for direction, target_class in self.inputs_spec:
            probe_point = (0, 0)
            vis_start = (0, 0)

            if direction == common.LEFT:
                probe_point = (x - common.PROBE_DISTANCE, center_y)
                vis_start = (x, center_y)
            elif direction == common.RIGHT:
                probe_point = (x + self.width + common.PROBE_DISTANCE, center_y)
                vis_start = (x + self.width, center_y)

            found_card = None
            vis_end = probe_point

            for candidate in active_cards:
                if candidate is not self and isinstance(candidate, target_class):
                    if candidate.is_point_inside(probe_point):
                        found_card = candidate
                        break

            if found_card:
                self.resolved_inputs[direction] = found_card
                self.input_ttls[direction] = common.INPUT_LINK_TTL
            else:
                if direction in self.input_ttls:
                    self.input_ttls[direction] -= 1
                    if self.input_ttls[direction] <= 0:
                        self.resolved_inputs.pop(direction, None)
                        del self.input_ttls[direction]
                else:
                    self.resolved_inputs.pop(direction, None)

            is_connected = direction in self.resolved_inputs
            self.conn_lines.append((vis_start, vis_end, is_connected))

    def get_priority(self):
        if not self.resolved_inputs: return 0
        max_input_prio = 0
        for direction, card in self.resolved_inputs.items():
            max_input_prio = max(max_input_prio, card.get_priority())
        return 1 + max_input_prio

    def project_output(self, direction, payload_card):
        x, y = self.top_left
        center_x = x + (self.width // 2)
        bottom_y = y + self.height

        if direction == common.DOWN:
            vx = center_x - (payload_card.width // 2)
            vy = y + common.VIRTUAL_CARD_OFFSET_Y
            target_card = payload_card

            if self.output_generated is not None:
                target_card = self.output_generated
                if hasattr(target_card, 'img_arr') and hasattr(payload_card, 'img_arr'):
                    target_card.img_arr = payload_card.img_arr
                elif hasattr(target_card, 'kernel_arr') and hasattr(payload_card, 'kernel_arr'):
                    target_card.kernel_arr = payload_card.kernel_arr
            else:
                self.output_generated = target_card
                CardSystem.register(target_card, pos=(vx, vy), is_virtual=True)

            if target_card in CardSystem._store:
                CardSystem._store[target_card]['pos'] = (vx, vy)

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
        common.project_image_at(CardSystem.canvas, img, self.top_left)

    def reset_logic_state(self):
        self.conn_lines = []

    def run_logic(self):
        pass

class ImageCard(BaseCard):
    def __init__(self, image_data):
        h, w = image_data.shape[:2] if image_data is not None else (0, 0)
        super().__init__(width=w, height=h, inputs=[], outputs=[])
        self.img_arr = image_data
        self.type = "source"

class KernelCard(BaseCard):
    def __init__(self, kernel_arr):
        super().__init__(width=200, height=200,
                         inputs=[(common.LEFT, ImageCard)],
                         outputs=[(common.DOWN, ImageCard)])
        self.type = "kernel"
        self.kernel_arr = kernel_arr

class KernelAdditionCard(BaseCard):
    def __init__(self):
        super().__init__(width=200, height=200,
                         inputs=[(common.LEFT, KernelCard), (common.RIGHT, KernelCard)],
                         outputs=[(common.DOWN, KernelCard)])
        self.type = "operation"

# --- FACTORY & HELPERS ---

CLASS_SCRIPTS = {
    ImageCard: common.IMAGE_SCRIPT,
    KernelCard: common.KERNEL_SCRIPT,
    KernelAdditionCard: common.ADDITION_SCRIPT
}

def load_card_library_json(path="cards.json"):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading card library: {e}")
        return {}

JSON_CARD_LIBRARY = load_card_library_json()

def create_physical_card_instance(marker_id):
    key = str(marker_id)
    if key not in JSON_CARD_LIBRARY:
        return None

    config = JSON_CARD_LIBRARY[key]
    init_script = config.get("init_script", "")
    if not init_script: return None

    local_scope = {
        "cv2": cv2,
        "np": np,
        "Image": Image,
        "ImageCard": ImageCard,
        "KernelCard": KernelCard,
        "KernelAdditionCard": KernelAdditionCard
    }

    try:
        exec(init_script, globals(), local_scope)
        return local_scope.get("card")
    except Exception as e:
        print(f"Error instantiating card {marker_id}: {e}")
        return None