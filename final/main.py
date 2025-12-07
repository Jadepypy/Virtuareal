import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
CANVAS_W, CANVAS_H = 1920, 1080
# Use 4x4 dictionary (best for detection speed on Pi)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
# IDs for the 4 corners of your whiteboard
ANCHOR_IDS = [0, 1, 2, 3]


# --- 1. LOAD IMAGES (The Lookup Database) ---
def load_image_resource(filename, width=200):
    """Loads an image and converts it to a format OpenCV can use."""
    try:
        pil_img = Image.open(filename).convert("RGB")  # Keep RGB for display
        # Resize logic to keep aspect ratio
        w_percent = (width / float(pil_img.size[0]))
        h_size = int((float(pil_img.size[1]) * float(w_percent)))
        pil_img = pil_img.resize((width, h_size), Image.Resampling.LANCZOS)

        # Convert to numpy array (OpenCV format)
        # Note: OpenCV uses BGR, PIL uses RGB, so we swap channels
        numpy_img = np.array(pil_img)
        return cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        # Return a fallback colored square
        fallback = np.zeros((200, 200, 3), dtype=np.uint8)
        fallback[:] = (0, 255, 0)  # Green
        return fallback


print("Loading Resources...")
# Map ArUco ID -> Image Data
IMAGE_DB = {
    10: load_image_resource("haruka.jpg", width=250),
    20: load_image_resource("blur3.png", width=200)
}


# --- 2. VISION HELPERS ---

def get_homography(corners, ids):
    """Calculates the map between Camera Space and Projector Space"""
    if ids is None: return None
    found_anchors = {}

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in ANCHOR_IDS:
            # Save the center of the anchor marker
            found_anchors[marker_id] = np.mean(corners[i][0], axis=0)

    # We need all 4 anchors to define the screen
    if len(found_anchors) < 4: return None

    # Source Points (Camera): TopLeft, TopRight, BtmRight, BtmLeft
    # Ensure this order matches your physical placement!
    src = np.array([
        found_anchors[0],
        found_anchors[1],
        found_anchors[2],
        found_anchors[3]
    ], dtype="float32")

    # Dest Points (Projector): TopLeft, TopRight, BtmRight, BtmLeft
    dst = np.array([
        [0, 0],
        [CANVAS_W, 0],
        [CANVAS_W, CANVAS_H],
        [0, CANVAS_H]
    ], dtype="float32")

    return cv2.getPerspectiveTransform(src, dst)


def transform_point(point, M):
    """Applies the Homography Matrix M to a single point"""
    pt = np.array([[[point[0], point[1]]]], dtype='float32')
    t = cv2.perspectiveTransform(pt, M)
    return int(t[0][0][0]), int(t[0][0][1])


# --- 3. MAIN LOOP ---

def main():
    cap = cv2.VideoCapture(0)
    # Set to 720p for Raspberry Pi performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE))

    print("System Started. Press 'q' to exit.")

    while True:
        # 1. READ POINTS (Capture)
        ret, frame = cap.read()
        if not ret: break

        corners, ids, rejected = detector.detectMarkers(frame)

        # 2. ANCHOR (Calibration)
        M = get_homography(corners, ids)

        # Initialize Black Canvas
        projector_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # 3. LOCATE CARDS & 4. LOOK UP IMAGES
        if ids is not None and M is not None:
            for i, marker_id in enumerate(ids.flatten()):

                # Check if this ID maps to an image
                if marker_id in IMAGE_DB:
                    # Get the image to draw
                    img_to_draw = IMAGE_DB[marker_id]

                    # --- ADJUSTED LOGIC: ANCHORING ---
                    # OpenCV Corners: [TopLeft, TopRight, BtmRight, BtmLeft]
                    # We want the Bottom-Left corner of the marker to be the start of the image
                    marker_corners = corners[i][0]
                    marker_btm_left = marker_corners[3]

                    # Transform to Projector coordinates
                    cx, cy = transform_point(marker_btm_left, M)

                    # 5. PROJECT IMAGE (Anchored at Top-Left of Image)
                    h, w, _ = img_to_draw.shape

                    # Draw extending Down and Right from the anchor
                    y1 = cy
                    y2 = cy + h
                    x1 = cx
                    x2 = cx + w

                    # Boundary checks (Don't crash if card goes off screen)
                    # We calculate the "valid slice" that fits on screen

                    # Canvas boundaries
                    c_y1 = max(0, y1);
                    c_y2 = min(CANVAS_H, y2)
                    c_x1 = max(0, x1);
                    c_x2 = min(CANVAS_W, x2)

                    # Image boundaries (matching the canvas slice)
                    # Note: We subtract the original (x1, y1) to find the offset in the source image
                    i_y1 = c_y1 - y1;
                    i_y2 = h - (y2 - c_y2)
                    i_x1 = c_x1 - x1;
                    i_x2 = w - (x2 - c_x2)

                    # Copy pixels only if dimensions are valid
                    if c_x2 > c_x1 and c_y2 > c_y1:
                        projector_canvas[c_y1:c_y2, c_x1:c_x2] = img_to_draw[i_y1:i_y2, i_x1:i_x2]

        # Draw Calibration Status (Visual Aid)
        if M is None:
            cv2.putText(projector_canvas, "CALIBRATING... FINDING 4 CORNERS", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Show Results
        cv2.imshow("Projector Output", projector_canvas)
        # Optional: Show camera feed for debugging
        # cv2.imshow("Debug Input", frame)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()