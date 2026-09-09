"""One-time: build static/img/board.png, a sharp transparent image of the
12x16 TB2 Mirror board with all four hold sets.

Aurora's Wood/Plastic base layers (37/38) are upscaled from a low-res source
and look blurry. Tension publishes a 4319px render of the 12x12 Mirror layout,
which is exactly those 498 base holds. This script:

  1. segments the holds out of that render's grey background,
  2. registers their centroids against the Aurora base layers (whose
     calibration is known) to find the render's pixel <-> board-unit mapping,
  3. composites the cut-out base holds with Aurora's sharp expansion layers
     (41/42) into one image spanning the full board edges.

    python make_board_image.py path/to/12x12_mirror_render.png
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# board edges for the 12x16 (product_size 10), what board.png spans
LEFT, RIGHT, BOTTOM, TOP = -92, 92, 0, 144
PX_PER_UNIT = 12                       # output resolution (Aurora is ~7.9)
OUT_W, OUT_H = (RIGHT - LEFT) * PX_PER_UNIT, (TOP - BOTTOM) * PX_PER_UNIT

BASE = ['static/img/37.png', 'static/img/38.png']
EXPANSION = ['static/img/41.png', 'static/img/42.png']


def blobs(mask, min_area):
    lab, n = ndimage.label(mask)
    areas = ndimage.sum(mask, lab, range(1, n + 1))
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    cents = ndimage.center_of_mass(mask, lab, keep)
    return np.array([(c[1], c[0]) for c in cents])  # (x, y)


# --- reference: Aurora base layers, centroids in board units ------------------
ref_alpha = None
for p in BASE:
    a = np.asarray(Image.open(p).convert('RGBA'))[..., 3]
    ref_alpha = a if ref_alpha is None else np.maximum(ref_alpha, a)
H, W = ref_alpha.shape
ref_px = blobs(ndimage.binary_fill_holes(ref_alpha > 128), 30)
ref_units = np.column_stack([LEFT + ref_px[:, 0] / W * (RIGHT - LEFT),
                             TOP - ref_px[:, 1] / H * (TOP - BOTTOM)])
print('aurora base blobs', len(ref_units))

# --- render: segment holds from the grey gradient background ------------------
src = Image.open(sys.argv[1]).convert('RGB')
rgb = np.asarray(src).astype(np.float32)
lum = rgb.mean(-1)
# background = wide median of a downscaled copy (holds are too small to survive)
small = np.asarray(src.convert('L').resize((src.width // 8, src.height // 8), Image.BOX)).astype(np.float32)
bg = ndimage.median_filter(small, size=45)
bg = np.asarray(Image.fromarray(bg).resize(src.size, Image.BILINEAR))
chroma = rgb.max(-1) - rgb.min(-1)
mask = (np.abs(lum - bg) > 28) | (chroma > 18)
mask[:int(src.height * 0.055)] = False             # title text above the board
mask = ndimage.binary_closing(mask, iterations=4)
mask = ndimage.binary_fill_holes(mask)
mask = ndimage.binary_opening(mask, iterations=4)
new_px = blobs(mask, 400)
print('render blobs', len(new_px))

# --- register: units -> render pixels, (px, py) = (sx*x + tx, -sy*y + ty) -----
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
for it in range(8):
    proj = np.column_stack([sx * ref_units[:, 0] + tx, -sy * ref_units[:, 1] + ty])
    d = np.linalg.norm(proj[:, None, :] - new_px[None, :, :], axis=2)
    j = d.argmin(1); dist = d[np.arange(len(proj)), j]
    ok = dist < 2.0 * sx                              # within 2 board units
    sx, tx, sy, ty = fit(ref_units[ok], new_px[j[ok]])
print(f'matched {ok.sum()}/{len(proj)}, median err {np.median(dist[ok]) / sx:.3f} units, '
      f'render is {sx:.1f} px/unit')

# --- cut out the base holds with a soft alpha edge ----------------------------
alpha = ndimage.gaussian_filter(mask.astype(np.float32), 1.0) * 255
cut = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), 'RGBA')

# --- composite into the output frame ------------------------------------------
out = Image.new('RGBA', (OUT_W, OUT_H), (0, 0, 0, 0))

# base holds: the render pixel box that maps onto the whole output frame
box = (sx * LEFT + tx, -sy * TOP + ty, sx * RIGHT + tx, -sy * BOTTOM + ty)
base = cut.transform((OUT_W, OUT_H), Image.EXTENT, box, Image.BICUBIC)
out.alpha_composite(base)          # areas outside the render come out transparent

for p in EXPANSION:
    layer = Image.open(p).convert('RGBA').resize((OUT_W, OUT_H), Image.LANCZOS)
    out.alpha_composite(layer)

out.save('static/img/board.png', optimize=True)
print('wrote static/img/board.png', out.size)
