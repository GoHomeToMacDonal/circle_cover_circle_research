"""Extract unit-circle centres from Friedman's pictures (v2).

Fixes: mask the big circle's own outline (which creates phantom Hough votes on a
ring of radius R+-1) and use a ring matched filter via FFT correlation.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter

R_TABLE = {
    19: np.sqrt(13),
    25: np.sqrt(6) + np.sqrt(3),
}


def load(path: str) -> np.ndarray:
    """Flatten onto a white background: the PNGs carry an alpha channel and a
    naive convert('L') turns transparent pixels black."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("L"), dtype=float) / 255.0


def disk_geometry(a: np.ndarray) -> tuple[float, float, float]:
    """Centre + radius of the shaded covered disk.

    The shaded disk is the only region that is solidly non-white, so we take the
    largest radius at which essentially every pixel is non-white.
    """
    grey = np.isclose(a, np.bincount((a * 255).astype(int).ravel())[:250].argmax() / 255.0,
                      atol=1 / 510)
    ys, xs = np.nonzero(grey)
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5

    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    notwhite = a < 250 / 255
    r = 1.0
    step = 0.25
    while r < min(h, w) / 2:
        m = (rr >= r) & (rr < r + step)
        if m.sum() > 0 and notwhite[m].mean() < 0.995:
            break
        r += step
    # the disk outline itself is ~1px wide; back it off
    return cx, cy, r - 0.5


def ring_correlate(dark: np.ndarray, radius: float, width: float = 1.6) -> np.ndarray:
    """Correlate the edge map with an annulus template of the given radius."""
    k = int(np.ceil(radius + width + 2))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    d = np.hypot(xx, yy)
    tmpl = np.clip(1.0 - np.abs(d - radius) / width, 0.0, 1.0)
    tmpl /= tmpl.sum()
    from scipy.signal import fftconvolve

    return fftconvolve(dark.astype(float), tmpl[::-1, ::-1], mode="same")


def nms_peaks(score: np.ndarray, n: int, min_sep: float):
    sm = gaussian_filter(score, 1.0)
    mx = maximum_filter(sm, size=3)
    cand = np.argwhere((sm == mx) & (sm > 0.15 * sm.max()))
    vals = sm[cand[:, 0], cand[:, 1]]
    order = np.argsort(vals)[::-1]
    out = []
    for i in order:
        y, x = cand[i]
        if all((x - px) ** 2 + (y - py) ** 2 >= min_sep**2 for px, py in out):
            out.append((float(x), float(y)))
        if len(out) >= n:
            break
    return out


def refine(dark: np.ndarray, pts, radius: float, width: float = 1.6):
    """Sub-pixel refine each centre by local quadratic fit on the ring score."""
    from scipy.optimize import minimize

    ys, xs = np.nonzero(dark)
    px, py = xs + 0.5, ys + 0.5

    def neg(p):
        d = np.hypot(px - p[0], py - p[1])
        w = np.clip(1.0 - np.abs(d - radius) / width, 0.0, None)
        return -w.sum()

    out = []
    for x, y in pts:
        res = minimize(neg, [x + 0.5, y + 0.5], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-3})
        out.append((res.x[0], res.x[1], -res.fun))
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    R = R_TABLE[n]
    a = load(f"data/{n}.png")
    cx, cy, Rpx = disk_geometry(a)
    upx = Rpx / R
    print(f"image {a.shape}  centre=({cx:.2f},{cy:.2f})  Rpx={Rpx:.2f}  unit={upx:.3f}px")

    dark = a < 0.45
    h, w = dark.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    masked = dark & (np.abs(rr - Rpx) > 2.5)
    print(f"dark {dark.sum()} -> masked {masked.sum()}")

    score = ring_correlate(masked, upx)
    pts = nms_peaks(score, n, min_sep=upx * 0.45)
    ref = refine(masked, pts, upx)

    rows = []
    for x, y, s in ref:
        X, Y = (x - cx) / upx, -(y - cy) / upx
        rows.append((X, Y, np.hypot(X, Y), np.degrees(np.arctan2(Y, X)) % 360, s))
    rows.sort(key=lambda t: (round(t[2], 1), t[3]))
    print("\n idx        x        y     rho    theta   score")
    for i, (X, Y, rho, th, s) in enumerate(rows):
        print(f" {i:3d} {X:8.4f} {Y:8.4f} {rho:7.4f} {th:8.2f} {s:7.1f}")
    arr = np.array([[X, Y] for X, Y, *_ in rows])
    np.save(f"data/centers_{n}.npy", arr)

    # overlay for visual check
    img = Image.open(f"data/{n}.png").convert("RGB")
    dr = ImageDraw.Draw(img)
    for x, y, _ in ref:
        dr.ellipse([x - upx, y - upx, x + upx, y + upx], outline=(255, 0, 0))
        dr.point((x, y), fill=(0, 0, 255))
    img.resize((600, 600), Image.NEAREST).save(f"data/overlay_{n}.png")
    print(f"\nwrote data/overlay_{n}.png")


if __name__ == "__main__":
    main()
