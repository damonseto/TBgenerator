"""One-time: turn the Tension 12x16 Mirror layout render into a calibrated,
transparent board image (static/img/board.png).

Holds are segmented from the grey background, then their centroids are
registered against the centroids from the official Aurora layer images
(whose calibration we already know) to find the pixel <-> board-unit mapping.

    python make_board_image.py ~/Downloads/12x16_Mirror.webp
"""
import json
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

LEFT, RIGHT, BOTTOM, TOP = -92, 92, 0, 144
AURORA = ['static/img/37.png', 'static/img/38.png', 'static/img/41.png', 'static/img/42.png']


def blobs(mask, min_area):
    lab, n = ndimage.label(mask)
    areas = ndimage.sum(mask, lab, range(1, n + 1))
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    cents = ndimage.center_of_mass(mask, lab, keep)
    return np.array([(c[1], c[0]) for c in cents])  # (x, y)


# --- reference: Aurora composite, centroids in board units --------------------
ref_alpha = np.zeros(Image.open(AURORA[0]).size[::-1])
for p in AURORA:
    ref_alpha = np.maximum(ref_alpha, np.asarray(Image.open(p).convert('RGBA'))[..., 3])
ref_mask = ndimage.binary_fill_holes(ref_alpha > 128)
H, W = ref_mask.shape
ref_px = blobs(ref_mask, 30)
ref_units = np.column_stack([LEFT + ref_px[:, 0] / W * (RIGHT - LEFT),
                             TOP - ref_px[:, 1] / H * (TOP - BOTTOM)])
print('aurora blobs', len(ref_units))

# --- new image: segment holds -------------------------------------------------
src = Image.open(sys.argv[1]).convert('RGB')
rgb = np.asarray(src).astype(float)
lum = rgb.mean(-1)
# The background is a smooth grey gradient (plus a faint logo). Estimate it
# with a wide median filter, which holds are too small to survive, then keep
# whatever differs from it in brightness or has any colour (grey has none).
bg = ndimage.median_filter(lum, size=75)
chroma = rgb.max(-1) - rgb.min(-1)
mask = (np.abs(lum - bg) > 22) | (chroma > 16)
mask[:110] = False                      # title text
mask = ndimage.binary_closing(mask, iterations=2)
mask = ndimage.binary_fill_holes(mask)
mask = ndimage.binary_opening(mask, iterations=1)
new_px = blobs(mask, 30)
print('new blobs', len(new_px))

# --- register: units -> pixels, (px, py) = (sx*x + tx, -sy*y + ty) ------------
# initial guess from bounding boxes
def fit(units, px):
    A = np.column_stack([units[:, 0], np.ones(len(units))])
    sx, tx = np.linalg.lstsq(A, px[:, 0], rcond=None)[0]
    B = np.column_stack([-units[:, 1], np.ones(len(units))])
    sy, ty = np.linalg.lstsq(B, px[:, 1], rcond=None)[0]
    return sx, tx, sy, ty

u_lo, u_hi = ref_units.min(0), ref_units.max(0)
p_lo, p_hi = new_px.min(0), new_px.max(0)
sx = (p_hi[0] - p_lo[0]) / (u_hi[0] - u_lo[0]); tx = p_lo[0] - sx * u_lo[0]
sy = (p_hi[1] - p_lo[1]) / (u_hi[1] - u_lo[1]); ty = p_hi[1] + sy * u_lo[1]

for it in range(10):
    proj = np.column_stack([sx * ref_units[:, 0] + tx, -sy * ref_units[:, 1] + ty])
    d = np.linalg.norm(proj[:, None, :] - new_px[None, :, :], axis=2)
    j = d.argmin(1); dist = d[np.arange(len(proj)), j]
    ok = dist < 12
    sx, tx, sy, ty = fit(ref_units[ok], new_px[j[ok]])
    print(f'iter {it}: matched {ok.sum()}/{len(proj)} median err {np.median(dist[ok]):.2f}px')

# --- crop to the board extent, transparent background -------------------------
x0, x1 = sx * LEFT + tx, sx * RIGHT + tx
y0, y1 = -sy * TOP + ty, -sy * BOTTOM + ty
print(f'board extent in image px: x {x0:.1f}..{x1:.1f}  y {y0:.1f}..{y1:.1f}')

soft = ndimage.binary_dilation(mask, iterations=1)
alpha = ndimage.gaussian_filter(soft.astype(float), 0.7) * 255
out = np.dstack([rgb, alpha]).astype(np.uint8)
im = Image.fromarray(out, 'RGBA')

# resample so the crop box lands exactly on integer pixels
box = (x0, y0, x1, y1)
w, h = int(round(x1 - x0)), int(round(y1 - y0))
im = im.transform((w, h), Image.EXTENT, box, Image.BICUBIC)
im.save('static/img/board.png')
print('wrote static/img/board.png', im.size)
