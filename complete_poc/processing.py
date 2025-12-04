import numpy as np
from scipy.signal import convolve2d


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class Card:
    def __init__(self, card_id, name, type="kernel", kernel=None, op_mode=None, source_image=None):
        self.id = card_id
        self.name = name
        self.type = type  # "source", "kernel", or "display"

        # Logic properties
        self.kernel = kernel
        self.op_mode = op_mode  # e.g., "abs" for gradients
        self.static_image = source_image  # For Source cards

        # State properties (Updated every frame by the engine)
        self.center = (0, 0)  # (x, y)
        self.output = None  # The image this card is currently holding/showing
        self.input_source = None  # Which card am I listening to?

    def set_position(self, x, y):
        self.center = (x, y)

    def find_left_neighbor(self, all_cards, max_dist=1000):
        """
        Looks for the closest card physically to the LEFT of this card.
        """
        my_x, my_y = self.center
        closest_card = None
        min_dist = max_dist

        for other_card in all_cards:
            if other_card.id == self.id: continue  # Don't find myself

            ox, oy = other_card.center
            print(ox - my_x, oy - my_y)

            # 1. Direction Check: Is it to the LEFT?
            # We define "Left" as having a smaller X value.
            # We also check Y to ensure it's roughly in the same "row" (y-diff < 150px)
            if ox < my_x and abs(oy - my_y) < 150:

                # 2. Distance Check
                dist = get_distance(self.center, other_card.center)
                if dist < min_dist:
                    min_dist = dist
                    closest_card = other_card

        return closest_card

    def compute(self, all_active_cards):
        """
        The Main Brain: Pulls input from neighbor, calculates output.
        """
        # CASE A: I am a Source Card (I generate data)
        if self.type == "source":
            self.output = self.static_image
            return

        # CASE B: I am a Kernel/Processor (I need input)
        neighbor = self.find_left_neighbor(all_active_cards)
        self.input_source = neighbor  # Remember who for drawing arrows later

        if neighbor and neighbor.output is not None:
            # 1. Grab Input
            input_img = neighbor.output

            # 2. Apply Kernel (Convolution)
            if self.kernel is not None:
                # Scipy convolve2d is accurate.
                # using 'same' ensures output size matches input size
                convolved = convolve2d(input_img, self.kernel, mode='same', boundary='symm')

                if self.op_mode == "abs":
                    self.output = np.abs(convolved)
                else:
                    self.output = convolved
            else:
                # Pass-through (if no kernel defined)
                self.output = input_img
        else:
            # No neighbor = No output (Card goes dark or shows waiting state)
            self.output = None