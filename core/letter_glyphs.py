"""Axis-aligned stencil letter glyphs for insert_letter (CoRL).

Slot order left→right: **C o R L** (lowercase ``o``, matching the CoRL acronym).

All geometry lives on a 2 mm integer grid so that:
  * peg strokes are exact boxes (zero convex-decomposition error);
  * pocket = inflate(strokes, CLEARANCE) is exact;
  * board body = tile_complement(board_rect, pockets) is an exact
    non-overlapping rectangle cover of (board \\ pockets).

Units: one grid unit ``u = 0.002`` m. Stroke rectangles are
``(x0, y0, x1, y1)`` in local letter frame with origin at the letter
center, x right, y forward (board +Y). Capitals span y∈[-13,13];
lowercase ``o`` is shorter and baseline-aligned (bottom at y=-13).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import math
import numpy as np

# ---------------------------------------------------------------------------
# Grid / physical constants
# ---------------------------------------------------------------------------
U = 0.002  # metres per grid unit

W = 20  # capital letter width in units  → 40 mm
H = 26  # capital letter height in units → 52 mm
S = 5   # stroke thickness (slightly thinner = cleaner silhouette)

PEG_H = 0.036       # peg height (m)
BOARD_H = 0.014     # single-layer board thickness (m) — no chamfer step
CLEARANCE = 0.0025  # radial clearance peg→pocket (m); no visual funnel needed

# Board layout: four through-holes in a row (CoRL order).
BOARD_W_U = 120  # 240 mm
BOARD_D_U = 40   # 80 mm
SLOT_SPACING_U = 28  # center-to-center (56 mm)
BOARD_Y = 0.14       # board center Y in world (m)

# Grasp: TCP height above table when holding the peg by its top.
GRASP_Z = 0.028

# CoRL acronym order (left → right on the board).
LETTERS = ("C", "o", "R", "L")

# RGBA per letter (distinct colours for Understanding).
PEG_COLORS: Dict[str, List[float]] = {
    "C": [0.90, 0.18, 0.18, 1.0],  # red
    "o": [0.18, 0.40, 0.90, 1.0],  # blue (lowercase)
    "R": [0.95, 0.78, 0.15, 1.0],  # yellow
    "L": [0.18, 0.72, 0.28, 1.0],  # green
}
BOARD_COLOR = [0.18, 0.18, 0.20, 1.0]  # slightly lighter charcoal

# Rotational symmetry (radians). Lowercase o has 180° symmetry.
SYMMETRY: Dict[str, float] = {
    "C": 0.0,
    "o": math.pi,
    "R": 0.0,
    "L": 0.0,
}

# Stroke = (x0, y0, x1, y1) in grid units, letter-local, origin at center.
Rect = Tuple[int, int, int, int]

GLYPHS: Dict[str, List[Rect]] = {
    # C — open on the right (capital).
    "C": [
        (-10, 8, 10, 13),     # top bar
        (-10, -8, -5, 8),     # left stem
        (-10, -13, 10, -8),   # bottom bar
    ],
    # o — smaller closed ring, baseline-aligned (reads as lowercase).
    # Capitals span [-13,13]; o spans [-13, 3] (x-height ≈ 16u).
    "o": [
        (-8, -2, 8, 3),       # top
        (-8, -13, 8, -8),     # bottom
        (-8, -8, -3, -2),     # left
        (3, -8, 8, -2),       # right
    ],
    # R — classic block R with a stepped leg (readable as R, not A/P).
    "R": [
        (-10, -13, -5, 13),   # left stem (full height)
        (-5, 8, 10, 13),      # top bar
        (5, 0, 10, 8),        # right upper (bowl side)
        (-5, -1, 10, 4),      # mid crossbar (bowl floor)
        (0, -7, 6, -1),       # leg upper step
        (4, -13, 10, -7),     # leg lower step (reads as diagonal)
    ],
    # L — left stem + bottom bar.
    "L": [
        (-10, -13, -5, 13),   # left stem
        (-5, -13, 10, -8),    # bottom bar
    ],
}


def normalize_letter(letter: str) -> str:
    """Map CLI / user input onto the canonical CoRL key (``o`` stays lower)."""
    s = str(letter).strip()
    if s in LETTERS:
        return s
    # Common aliases
    if s in ("O", "0"):
        return "o"
    up = s.upper()
    if up in ("C", "R", "L"):
        return up
    raise ValueError(f"target_letter must be one of {LETTERS} (CoRL), got {letter!r}")


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy / python, no sapien)
# ---------------------------------------------------------------------------

def rect_to_meters(r: Rect) -> Tuple[float, float, float, float]:
    """Grid rect → metres (x0, y0, x1, y1)."""
    return (r[0] * U, r[1] * U, r[2] * U, r[3] * U)


def inflate(rects: Sequence[Rect], clearance_m: float) -> List[Tuple[float, float, float, float]]:
    """Expand each rect outward by ``clearance_m`` on all four sides (metres)."""
    c = float(clearance_m)
    out = []
    for r in rects:
        x0, y0, x1, y1 = rect_to_meters(r)
        out.append((x0 - c, y0 - c, x1 + c, y1 + c))
    return out


def merge_rects(rects: Sequence[Tuple[float, float, float, float]],
                tol: float = 1e-9) -> List[Tuple[float, float, float, float]]:
    """Greedy axis-aligned merge of overlapping / abutting rectangles."""
    if not rects:
        return []
    cur = [tuple(map(float, r)) for r in rects]

    def overlaps_or_abuts(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (ax0 <= bx1 + tol and bx0 <= ax1 + tol
                and ay0 <= by1 + tol and by0 <= ay1 + tol)

    changed = True
    while changed:
        changed = False
        nxt = []
        used = [False] * len(cur)
        for i in range(len(cur)):
            if used[i]:
                continue
            ax0, ay0, ax1, ay1 = cur[i]
            used[i] = True
            grew = True
            while grew:
                grew = False
                for j in range(len(cur)):
                    if used[j]:
                        continue
                    b = cur[j]
                    if not overlaps_or_abuts((ax0, ay0, ax1, ay1), b):
                        continue
                    bx0, by0, bx1, by1 = b
                    ux0, uy0 = min(ax0, bx0), min(ay0, by0)
                    ux1, uy1 = max(ax1, bx1), max(ay1, by1)
                    area_u = (ux1 - ux0) * (uy1 - uy0)
                    area_a = (ax1 - ax0) * (ay1 - ay0)
                    area_b = (bx1 - bx0) * (by1 - by0)
                    ox0, oy0 = max(ax0, bx0), max(ay0, by0)
                    ox1, oy1 = min(ax1, bx1), min(ay1, by1)
                    area_o = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
                    if abs(area_u - (area_a + area_b - area_o)) < tol:
                        ax0, ay0, ax1, ay1 = ux0, uy0, ux1, uy1
                        used[j] = True
                        grew = True
                        changed = True
            nxt.append((ax0, ay0, ax1, ay1))
        cur = nxt
    return cur


def _pocket_mask(board_xu0: int, board_yu0: int, board_xu1: int, board_yu1: int,
                 pockets_u: Sequence[Sequence[Rect]]) -> np.ndarray:
    """Boolean mask over grid cells: True where a pocket occupies the cell."""
    nx = board_xu1 - board_xu0
    ny = board_yu1 - board_yu0
    mask = np.zeros((nx, ny), dtype=bool)
    for pocket in pockets_u:
        for x0, y0, x1, y1 in pocket:
            ix0 = max(board_xu0, x0) - board_xu0
            iy0 = max(board_yu0, y0) - board_yu0
            ix1 = min(board_xu1, x1) - board_xu0
            iy1 = min(board_yu1, y1) - board_yu0
            if ix1 > ix0 and iy1 > iy0:
                mask[ix0:ix1, iy0:iy1] = True
    return mask


def tile_complement(
    board_rect_u: Rect,
    pockets_local_u: Sequence[Sequence[Rect]],
) -> List[Rect]:
    """Exact rectangle tiling of ``board_rect_u \\ union(pockets)``."""
    bx0, by0, bx1, by1 = board_rect_u
    mask = _pocket_mask(bx0, by0, bx1, by1, pockets_local_u)
    free = ~mask
    nx, ny = free.shape
    visited = np.zeros_like(free, dtype=bool)
    tiles: List[Rect] = []

    for j in range(ny):
        for i in range(nx):
            if visited[i, j] or not free[i, j]:
                continue
            w = 1
            while i + w < nx and free[i + w, j] and not visited[i + w, j]:
                w += 1
            h = 1
            while j + h < ny:
                if not free[i:i + w, j + h].all():
                    break
                if visited[i:i + w, j + h].any():
                    break
                h += 1
            visited[i:i + w, j:j + h] = True
            tiles.append((bx0 + i, by0 + j, bx0 + i + w, by0 + j + h))
    return tiles


def stroke_half_extents_m(r: Rect) -> Tuple[np.ndarray, np.ndarray]:
    """Return (center_xy, half_size_xy) in metres for a grid stroke rect."""
    x0, y0, x1, y1 = rect_to_meters(r)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0)
    hy = 0.5 * (y1 - y0)
    return np.array([cx, cy], dtype=np.float64), np.array([hx, hy], dtype=np.float64)


def grasp_stroke(letter: str) -> Rect:
    """Pick the tallest leftmost stem for a stable pinch."""
    letter = normalize_letter(letter)
    strokes = GLYPHS[letter]
    best = None
    best_score = -1.0
    for r in strokes:
        x0, y0, x1, y1 = r
        dx, dy = x1 - x0, y1 - y0
        score = abs(dy) * 10.0 + abs(dx) * abs(dy) * 0.01
        score -= x0 * 0.001
        if score > best_score:
            best_score = score
            best = r
    assert best is not None
    return best


def slot_centers_world() -> Dict[str, np.ndarray]:
    """World-frame (x, y, z=0) of each letter slot center (CoRL left→right)."""
    n = len(LETTERS)
    span = (n - 1) * SLOT_SPACING_U * U
    x0 = -0.5 * span
    out = {}
    for i, L in enumerate(LETTERS):
        out[L] = np.array([x0 + i * SLOT_SPACING_U * U, BOARD_Y, 0.0],
                          dtype=np.float64)
    return out


def board_rect_u() -> Rect:
    """Board footprint in board-local grid units (origin = board center)."""
    hw, hd = BOARD_W_U // 2, BOARD_D_U // 2
    return (-hw, -hd, hw, hd)


def pockets_on_board_u(clearance_u: int = 1) -> List[List[Rect]]:
    """Inflated letter pockets in board-local grid units.

    ``clearance_u`` expands each stroke by that many units per side.
    Default 1u (= 2 mm) matches ``CLEARANCE`` closely; slight float
    CLEARANCE is applied at collision time via inflate() for pegs — board
    tiling uses integer expansion so the cover stays exact on the grid.
    """
    centers = slot_centers_world()
    board_origin_xy = np.array([0.0, BOARD_Y])
    pockets = []
    for L in LETTERS:
        cx_w, cy_w = centers[L][:2]
        cx_u = int(round((cx_w - board_origin_xy[0]) / U))
        cy_u = int(round((cy_w - board_origin_xy[1]) / U))
        # Use ceil(CLEARANCE/U) so the carved pocket is never tighter than
        # the physical clearance used when inflating peg strokes.
        expand = max(clearance_u, int(math.ceil(CLEARANCE / U - 1e-9)))
        local = []
        for r in GLYPHS[L]:
            x0, y0, x1, y1 = r
            local.append((
                x0 - expand + cx_u,
                y0 - expand + cy_u,
                x1 + expand + cx_u,
                y1 + expand + cy_u,
            ))
        pockets.append(local)
    return pockets


def board_tiles() -> List[Rect]:
    """Single-layer rectangle tiles for the solid board body."""
    return tile_complement(board_rect_u(), pockets_on_board_u())


def peg_bbox_m(letter: str) -> Tuple[float, float]:
    """Axis-aligned half-extents (hx, hy) of the full peg footprint (m)."""
    strokes = GLYPHS[normalize_letter(letter)]
    xs = [r[0] for r in strokes] + [r[2] for r in strokes]
    ys = [r[1] for r in strokes] + [r[3] for r in strokes]
    return (0.5 * (max(xs) - min(xs)) * U, 0.5 * (max(ys) - min(ys)) * U)
