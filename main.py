import cv2
import numpy as np
import common
import card_engine

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(common.ARUCO_DICT_TYPE))

    cv2.namedWindow(common.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(common.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("System Started. Press 'q' to exit. Press 'r' to reset calibration.")

    physical_cards_map = {}
    anchor_history = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in common.ANCHOR_IDS:
                    center = np.mean(corners[i][0], axis=0)
                    anchor_history[marker_id] = center

        M = common.get_homography_from_history(anchor_history)
        M_inv = np.linalg.inv(M) if M is not None else None

        projector_canvas = np.zeros((common.CANVAS_H, common.CANVAS_W, 3), dtype=np.uint8)

        # 1. SETUP CONTEXT
        card_engine.CardSystem.reset_frame(projector_canvas)

        # --- DEBUG VISUALIZATION ---
        if common.DEBUG_MODE:
            if corners is not None and ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for aid, pt in anchor_history.items():
                pt_int = tuple(pt.astype(int))
                cv2.circle(frame, pt_int, 12, (0, 255, 0), 2)
                cv2.putText(frame, f"A-{aid}", (pt_int[0] + 15, pt_int[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- UPDATE PHYSICAL CARDS ---
        seen_marker_ids = set()
        if ids is not None and M is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in common.ANCHOR_IDS:
                    mid = int(marker_id)
                    seen_marker_ids.add(mid)

                    marker_corners = corners[i][0]
                    marker_btm_left = marker_corners[3]
                    cx, cy = common.transform_point(marker_btm_left, M)

                    if mid not in physical_cards_map:
                        new_card = card_engine.create_physical_card_instance(mid)
                        if new_card:
                            physical_cards_map[mid] = new_card
                            card_engine.CardSystem.register(new_card, id=str(mid), pos=(cx, cy), is_virtual=False)

                    card = physical_cards_map.get(mid)
                    if card:
                        card_engine.CardSystem.update_physical(card, (cx, cy))

        keys_to_remove = []
        for mid, card in physical_cards_map.items():
            if mid not in seen_marker_ids:
                card_engine.CardSystem.decrease_ttl(card)
                if card_engine.CardSystem.get_ttl(card) <= 0:
                    keys_to_remove.append(mid)

        for k in keys_to_remove:
            card = physical_cards_map[k]
            card_engine.CardSystem.unregister(card)
            del physical_cards_map[k]

        # --- UNIFIED LOGIC LOOP ---
        active_cards = list(physical_cards_map.values())
        for c in active_cards: c.reset_logic_state()

        dummy_canvas = np.zeros_like(projector_canvas)

        # 2. SIMULATION PHASE
        for depth in range(common.MAX_CHAIN_DEPTH):
            for card in active_cards:
                card.resolve_dependencies(active_cards)

            active_cards.sort(key=lambda c: c.get_priority())

            card_engine.CardSystem.canvas = dummy_canvas
            new_virtuals = []
            for card in active_cards:
                card.run_logic()
                if card.output_generated:
                    if card.output_generated not in active_cards and card.output_generated not in new_virtuals:
                        new_virtuals.append(card.output_generated)

            if not new_virtuals:
                break
            active_cards.extend(new_virtuals)

        # 3. RENDER PHASE (Final Draw)
        card_engine.CardSystem.canvas = projector_canvas

        for card in active_cards:
            card.resolve_dependencies(active_cards)
            card.run_logic()
            card.draw_connections()

        # --- DEBUG OVERLAY ---
        if common.DEBUG_MODE and M_inv is not None:
            for card in active_cards:
                for start_pt, end_pt, is_connected in card.conn_lines:
                    cam_tip = common.transform_point(end_pt, M_inv)
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

        cv2.imshow(common.WINDOW_NAME, projector_canvas)
        if common.DEBUG_MODE:
            cv2.imshow("Debug Input", frame)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        if key == ord('r'):
            anchor_history.clear()
            print("Calibration Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()