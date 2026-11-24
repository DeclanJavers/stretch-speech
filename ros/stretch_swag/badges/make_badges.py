"""Print-ready ArUco badges for visitors (the perception cheat).

    python3 badges/make_badges.py                 # badges 100-109, 100 mm markers, A4/Letter pages

Stretch's detector uses the 6x6_250 dictionary. After printing, MEASURE the black square
and put that size in config/badges.yaml (length_mm): the detector uses it for distance.
"""
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DPI = 300
MM = DPI / 25.4
MARKER_MM, IDS = 100, range(100, 110)

aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
badges = []
for i in IDS:
    side = int(MARKER_MM * MM)
    marker = cv2.aruco.generateImageMarker(aruco, i, side)
    pad = int(8 * MM)                                   # white quiet zone around the marker
    card = np.full((side + 2 * pad + int(10 * MM), side + 2 * pad), 255, np.uint8)
    card[pad:pad + side, pad:pad + side] = marker
    cv2.putText(card, f"Stretch badge {i}", (pad, pad + side + int(7 * MM)), cv2.FONT_HERSHEY_SIMPLEX,
                2.2, 0, 4, cv2.LINE_AA)
    cv2.rectangle(card, (0, 0), (card.shape[1] - 1, card.shape[0] - 1), 180, 2)   # cut line
    badges.append(card)

# Two badges per page (fits A4 and US Letter at 300 dpi), saved as PNG and as one PDF.
page_w, page_h = int(210 * MM), int(279 * MM)
pages = []
for k in range(0, len(badges), 2):
    page = np.full((page_h, page_w), 255, np.uint8)
    for j, card in enumerate(badges[k:k + 2]):
        y = int(6 * MM) + j * (card.shape[0] + int(6 * MM))
        x = (page_w - card.shape[1]) // 2
        page[y:y + card.shape[0], x:x + card.shape[1]] = card
    pages.append(page)
    cv2.imwrite(str(HERE / f"badges_{IDS[k]}_{IDS[k] + 1}.png"), page)
print(f"Wrote {len(pages)} pages to {HERE} (print at 100% scale, no 'fit to page')")
