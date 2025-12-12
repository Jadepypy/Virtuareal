import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.units import mm
import os

# --- CONFIGURATION ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Dimensions
# A4 Size: 210mm x 297mm
# We use a 4:3 Aspect Ratio for the Image Card (240mm x 180mm)
# This fits perfectly on A4 Landscape (297mm wide)
CARDS = [
    # PAGE 1: Anchors (Standard Portrait)
    {"id": 0, "type": "anchor", "name": "TL Anchor", "width": 60, "height": 60},
    {"id": 1, "type": "anchor", "name": "TR Anchor", "width": 60, "height": 60},
    {"id": 2, "type": "anchor", "name": "BR Anchor", "width": 60, "height": 60},
    {"id": 3, "type": "anchor", "name": "BL Anchor", "width": 60, "height": 60},

    # PAGE 2: The "Cinema" Image Card (Huge)
    # 240mm x 180mm = 4:3 Aspect Ratio
    {"id": 10, "type": "image", "name": "Image Canvas (4:3)", "width": 240, "height": 180},

    # PAGE 3: The Kernel Card
    {"id": 20, "type": "kernel", "name": "Kernel Tool", "width": 120, "height": 160},
]

MARKER_SIZE_MM = 35
PADDING_MM = 5
GAP_MM = 10


def generate_marker_image(aruco_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    img = cv2.aruco.generateImageMarker(aruco_dict, aruco_id, 200)
    filename = f"temp_marker_{aruco_id}.png"
    cv2.imwrite(filename, img)
    return filename


def create_pdf(filename):
    # Start in Portrait
    c = canvas.Canvas(filename, pagesize=portrait(A4))

    # Initial Page Setup
    current_orientation = 'portrait'
    page_w, page_h = A4
    margin = 15 * mm

    x = margin
    y = page_h - margin
    max_row_h = 0

    print(f"Generating PDF: {filename}...")

    for card in CARDS:
        w = card['width'] * mm
        h = card['height'] * mm

        # --- SMART ROTATION LOGIC ---
        # 1. Check if card is too wide for Portrait
        # Printable width in Portrait is ~180mm. Our card is 240mm.
        needs_landscape = w > (210 * mm - 2 * margin)

        # 2. Check if we need to switch page orientation
        if needs_landscape and current_orientation == 'portrait':
            c.showPage()  # End current page
            c.setPageSize(landscape(A4))  # Switch to Landscape
            current_orientation = 'landscape'
            page_w, page_h = landscape(A4)  # Update dimensions
            x, y = margin, page_h - margin  # Reset cursor
            max_row_h = 0

        elif not needs_landscape and current_orientation == 'landscape':
            # Switch back to portrait if we were in landscape (optional, but keeps flow tidy)
            c.showPage()
            c.setPageSize(portrait(A4))
            current_orientation = 'portrait'
            page_w, page_h = portrait(A4)
            x, y = margin, page_h - margin
            max_row_h = 0

        # --- FLOW CHECK ---
        # Does it fit on the current line?
        if x + w > page_w - margin:
            x = margin
            y -= (max_row_h + GAP_MM * mm)
            max_row_h = 0

        # Does it fit on the current page?
        if y - h < margin:
            c.showPage()
            # Maintain current orientation for new page
            y = page_h - margin
            x = margin
            max_row_h = 0

        # --- DRAW CARD ---
        # Draw Cut Line
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(1)
        c.rect(x, y - h, w, h)

        # Draw Label
        c.setFont("Helvetica", 10)
        c.drawString(x, y + 2 * mm, f"ID: {card['id']} ({card['name']})")

        # Draw ArUco (Top Left)
        marker_file = generate_marker_image(card['id'])
        marker_x = x + (PADDING_MM * mm)
        marker_y = y - (PADDING_MM * mm) - (MARKER_SIZE_MM * mm)
        c.drawImage(marker_file, marker_x, marker_y, width=MARKER_SIZE_MM * mm, height=MARKER_SIZE_MM * mm)

        # Draw Projection Area Helper
        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setDash(4, 2)
        c.rect(x + 5 * mm,
               y - h + 5 * mm,
               w - 10 * mm,
               h - (MARKER_SIZE_MM * mm) - 15 * mm)
        c.setDash(1, 0)

        # Update Cursors
        if h > max_row_h: max_row_h = h
        x += w + (GAP_MM * mm)

        try:
            os.remove(marker_file)
        except:
            pass

    c.save()
    print("Done! Note: Check printer settings to handle Landscape pages automatically.")


if __name__ == "__main__":
    create_pdf("realtalk_cards.pdf")