"""Print-ready labels for the goodie bags: item name + its ArUco tag (50 mm), 6 labels per page.

    python3 badges/make_bag_labels.py          # -> badges/bag_labels.png (Letter/A4 at 300 dpi)

Tags: pen 200, shirt 201, plushie 202 (2 labels of each). Print at 100%, MEASURE the black
square, set length_mm for 200-202 in config/badges.yaml. Stick the label high on the bag's
front, centred, with the tag ~6-7 cm below the bag's top edge (pick.py assumes the loops' top
is 14.5 cm above the tag's centre: measure yours and set handle_above_tag_m).
"""
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DPI = 300
MM = DPI / 25.4
TAG_MM = 50
ITEMS = [("PEN", 200), ("T-SHIRT", 201), ("PLUSHIE", 202)] * 2

aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
labels = []
for name, tag in ITEMS:
    side = int(TAG_MM * MM)
    pad = int(7 * MM)
    w, h = side + 2 * pad, side + 2 * pad + int(14 * MM)
    card = np.full((h, w), 255, np.uint8)
    card[pad:pad + side, pad:pad + side] = cv2.aruco.generateImageMarker(aruco, tag, side)
    cv2.putText(card, name, (pad, pad + side + int(12 * MM)), cv2.FONT_HERSHEY_DUPLEX, 2.6, 0, 6)
    cv2.putText(card, f"#{tag}", (w - pad - int(14 * MM), pad - int(3 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.rectangle(card, (0, 0), (w - 1, h - 1), 180, 2)               # cut line
    labels.append(card)

page = np.full((int(279 * MM), int(210 * MM)), 255, np.uint8)        # fits Letter and A4
lh, lw = labels[0].shape
x0, y0 = int(15 * MM), int(10 * MM)
for k, card in enumerate(labels):
    r, c = divmod(k, 2)
    y, x = y0 + r * (lh + int(5 * MM)), x0 + c * (lw + int(10 * MM))
    page[y:y + lh, x:x + lw] = card
cv2.imwrite(str(HERE / "bag_labels.png"), page)
print("wrote", HERE / "bag_labels.png")
