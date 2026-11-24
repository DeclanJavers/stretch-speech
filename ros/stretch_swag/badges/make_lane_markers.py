"""Print-ready lane markers for the goodie-bag table (one per lane, behind its last bag).

    python3 badges/make_lane_markers.py        # -> badges/lane_markers.pdf (3 Letter pages, 300 dpi)

Print at 100% / "actual size" (not "fit to page"), then MEASURE the black square: it should be
100 mm. If it isn't, pass the measured size to the robot: marker_mm:=<mm>. Each page also has a
100 mm check bar. Mount each marker upright on a stick or a box behind its lane, facing the front
of the table, its bottom edge ~15 cm above the bags' loops (see STRETCH-TEST.md).
"""
import pathlib

import cv2
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
DPI = 300
MM = DPI / 25.4
MARKER_MM = 100
LANES = [(20, "PEN lane (left)"), (21, "T-SHIRT lane (middle)"), (22, "PLUSHIE lane (right)")]

aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
pages = []
for marker_id, label in LANES:
    page = np.full((int(11 * DPI), int(8.5 * DPI)), 255, np.uint8)      # US Letter
    side = int(round(MARKER_MM * MM))
    marker = cv2.aruco.generateImageMarker(aruco, marker_id, side)
    x0, y0 = (page.shape[1] - side) // 2, int(40 * MM)
    page[y0:y0 + side, x0:x0 + side] = marker
    pad = int(15 * MM)                                                  # keep this white margin when cutting
    cv2.rectangle(page, (x0 - pad, y0 - pad), (x0 + side + pad, y0 + side + pad), 170, 2)
    cv2.putText(page, label, (x0 - pad, y0 + side + pad + int(14 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 6,
                cv2.LINE_AA)
    cv2.putText(page, f"ArUco 6x6 id {marker_id}  -  black square must measure {MARKER_MM} mm", (x0 - pad, y0 + side + pad + int(24 * MM)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 60, 3, cv2.LINE_AA)
    cv2.putText(page, "Cut along the grey line (keep the white border). Face the front of the table.",
                (x0 - pad, y0 + side + pad + int(32 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 60, 2, cv2.LINE_AA)
    by = int(230 * MM)                                                  # 100 mm print-scale check bar
    bx = (page.shape[1] - side) // 2
    page[by:by + int(4 * MM), bx:bx + side] = 0
    cv2.putText(page, "this bar = 100 mm when printed at actual size", (bx, by + int(12 * MM)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 60, 3, cv2.LINE_AA)
    pages.append(Image.fromarray(page))

out = HERE / "lane_markers.pdf"
pages[0].save(out, save_all=True, append_images=pages[1:], resolution=DPI)
print(f"wrote {out}")
