import cv2
import numpy as np

# --- CONFIGURATION (Must match the generator) ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# The Database (Simulated JSON)
CARD_DB = {
    10: {"type": "image", "width": 160, "height": 120, "color": (0, 255, 255)},  # Yellow Image
    11: {"type": "image", "width": 160, "height": 120, "color": (0, 255, 0)},  # Green Image
    20: {"type": "kernel", "width": 100, "height": 140, "color": (255, 0, 255)},  # Magenta Kernel
    21: {"type": "kernel", "width": 100, "height": 140, "color": (0, 0, 255)},  # Red Kernel
}

# Physical Setup Constants
MARKER_SIZE_MM = 35
PADDING_MM = 5

# The Virtual Canvas Size (We map the whiteboard to this resolution)
CANVAS_W = 1920
CANVAS_H = 1080


def get_homography(corners, ids):
    """
    Finds the 4 anchor markers (0,1,2,3) and calculates the transformation matrix
    to map the whiteboard to the 1920x1080 Canvas.
    """
    if ids is None: return None

    # Create a simple dict for easier lookup
    found_anchors = {}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in [0, 1, 2, 3]:
            # Get center of the marker
            c = np.mean(corners[i][0], axis=0)
            found_anchors[marker_id] = c

    # We need all 4 anchors to establish the "World"
    if len(found_anchors) < 4:
        return None

    # Source Points: Where the anchors are in the CAMERA
    src_pts = np.array([
        found_anchors[0],  # Top Left
        found_anchors[1],  # Top Right
        found_anchors[2],  # Bottom Right
        found_anchors[3]  # Bottom Left
    ], dtype="float32")

    # Destination Points: Where we WANT them on the PROJECTOR/CANVAS
    dst_pts = np.array([
        [0, 0],
        [CANVAS_W, 0],
        [CANVAS_W, CANVAS_H],
        [0, CANVAS_H]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M


def transform_point(point, M):
    """Applies the Homography Matrix M to a single point (x,y)"""
    # Point must be shaped (1, 1, 2) for OpenCV
    pt_array = np.array([[[point[0], point[1]]]], dtype='float32')
    transformed = cv2.perspectiveTransform(pt_array, M)
    return transformed[0][0]  # Returns (x, y)


# flow:
# camera read -> detect markers -> find homography -> for each card: find position
# for each card: draw on canvas

def main():
    cap = cv2.VideoCapture(0)  # Logic's Camera

    # 1. Setup ArUco
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Scale factor: pixels per mm on the virtual canvas
    # Approx: 1920 pixels / 1200mm whiteboard = 1.6 px/mm
    px_per_mm = 1.6

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Detect Markers
        corners, ids, rejected = detector.detectMarkers(frame)

        # 3. Output Canvas (The Projector Simulator)
        # Start with Black (Light off)
        projector_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # 4. Calibration (Find the "World")
        M = get_homography(corners, ids)

        if M is not None:
            # --- WORLD IS LOCKED ---

            # (Optional) Visualize the distorted whiteboard in the Debug window
            warped_camera = cv2.warpPerspective(frame, M, (CANVAS_W, CANVAS_H))

            # 5. Locate Cards
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    mid = int(marker_id)

                    # Ignore anchors, we only care about "Cards"
                    if mid not in CARD_DB: continue

                    card_data = CARD_DB[mid]

                    # Get the Top-Left corner of the ARUCO marker in Camera Space
                    # corners[i][0] is [TL, TR, BR, BL] of the marker
                    marker_tl_camera = corners[i][0][0]

                    # Convert to Canvas Space
                    marker_tl_canvas = transform_point(marker_tl_camera, M)

                    # 6. Calculate Card Position
                    # We know the Marker is padded from the card's real top-left.
                    # Math: Card_TL = Marker_TL - Padding
                    # Note: This simple subtraction assumes the card isn't rotated much.
                    # For full rotation support, we'd need vector math, but this works for sliding.

                    pad_px = PADDING_MM * px_per_mm
                    card_x = int(marker_tl_canvas[0] - pad_px)
                    card_y = int(marker_tl_canvas[1] - pad_px)

                    card_w_px = int(card_data['width'] * px_per_mm)
                    card_h_px = int(card_data['height'] * px_per_mm)

                    # 7. Draw the "Projection"
                    # This is what simulates projecting an image BACK onto the card

                    # A. Draw the Card Background (The "Valid Projection Area")
                    cv2.rectangle(projector_canvas,
                                  (card_x, card_y),
                                  (card_x + card_w_px, card_y + card_h_px),
                                  card_data['color'], -1)  # Filled

                    # B. Draw Text (Simulating Content)
                    text = f"{card_data['type'].upper()} ID:{mid}"
                    cv2.putText(projector_canvas, text, (card_x + 10, card_y + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    # C. Draw the "Actual" ArUco location (for debug)
                    cv2.circle(projector_canvas, (int(marker_tl_canvas[0]), int(marker_tl_canvas[1])), 5,
                               (255, 255, 255), -1)

            # Show the warped camera view (Debug)
            cv2.imshow('Debug: Warped Camera (Homography)', cv2.resize(warped_camera, (640, 360)))

        else:
            # If anchors are missing, warn the user
            cv2.putText(projector_canvas, "ANCHORS NOT FOUND", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

        # 8. Show Results
        cv2.imshow('Camera Input', frame)
        # This is the important window: Full Screen this on your Projector!
        cv2.imshow('Projector Output (The Simulation)', projector_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()