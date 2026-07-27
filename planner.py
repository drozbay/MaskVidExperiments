"""Subject crop planner: shape as an optimization output.

Standalone numpy/scipy planner solved against binary masks; produces the
per-frame crop boxes only.

Structure per mode:
  tracked  -- one LP: scalar W, H + per-frame positions. Shape emerges.
  zoomed   -- outer sweep over constant aspect candidates, inner LP per
              candidate: per-frame height (width slaved to aspect) +
              positions. Rent is charged against the tightest
              aspect-respecting padded box, never per axis.
  combined -- closed form: padded union box (also the movement_cost -> inf
              limit of the other two).

Dials (p):
  crop_scale                     padding promise as a multiple of the subject
  min_padding_allowed            fraction of the promise enforced as hard floors
  min_padding_allowed_window     odd median window defining sustained motion
  pad_deficit_tol                shortfall above the floor costs 1/this per px-frame
  pad_surplus_tol                size beyond the padded need costs 1/this per px-frame
  resize_cost                    cost per pixel of box size change (zoomed)
  movement_cost                  cost per pixel of box movement
  center_pull                    tie-break pull toward the subject center
  end_tightening                 fraction of rent charged on the phantoms
  end_tightening_window          phantom frames held at each end
  zoom_step                      >1 quantizes sizes to geometric levels (zoomed)
  max_zoom_rate                  >1 caps per-frame size ratio change (zoomed), 0/1 off
  aspect_ratio                   >0 fixes the shape and skips the sweep
  seamless_loop                  periodic plan (positions and sizes wrap)
"""

import math

import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import linprog


def _extent_tracks(binary):
    """Per-frame subject extents, NaN gaps interpolated and edge-held."""
    n = binary.shape[0]
    bx0 = np.full(n, np.nan)
    bx1 = np.full(n, np.nan)
    by0 = np.full(n, np.nan)
    by1 = np.full(n, np.nan)
    for i, frame in enumerate(binary):
        ys, xs = frame.nonzero()
        if len(xs):
            bx0[i], bx1[i] = xs.min(), xs.max()
            by0[i], by1[i] = ys.min(), ys.max()
    known = ~np.isnan(bx0)
    idx = np.arange(n)
    out = [np.interp(idx, idx[known], t[known]) for t in (bx0, bx1, by0, by1)]
    return out, known


def _median_tracks(tracks, window):
    g = max(int(window), 1) | 1
    return [ndimage.median_filter(t, size=g, mode="nearest") for t in tracks]


def _round_corridor(t, lo, hi):
    ilo = np.ceil(lo - 1e-6)
    ihi = np.floor(hi + 1e-6)
    x = np.clip(np.round(t), ilo, ihi)
    return np.where(ilo > ihi, np.round((lo + hi) / 2), x).astype(int)


class _LP:
    """Row-wise builder for sparse linprog problems."""

    def __init__(self):
        self.blocks = []       # (name, size)
        self.off = {}
        self.nv = 0
        self.ri, self.ci, self.vv, self.rhs = [], [], [], []
        self.c = []
        self.lo, self.hi = [], []

    def var(self, name, size, cost=0.0, lo=0.0, hi=None):
        self.off[name] = self.nv
        self.nv += size
        cost = np.broadcast_to(np.asarray(cost, dtype=float), (size,))
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (size,))
        self.c.extend(cost)
        self.lo.extend(lo)
        if hi is None:
            self.hi.extend([None] * size)
        else:
            hi_arr = np.broadcast_to(np.asarray(hi, dtype=float), (size,))
            self.hi.extend(hi_arr)
        return self.off[name]

    def row(self, cols, vals, b):
        r = len(self.rhs)
        self.ri.extend([r] * len(cols))
        self.ci.extend(cols)
        self.vv.extend(vals)
        self.rhs.append(b)

    def solve(self, eq=None):
        A = sparse.csr_matrix((self.vv, (self.ri, self.ci)),
                              shape=(len(self.rhs), self.nv))
        bounds = [(l, h) for l, h in zip(self.lo, self.hi)]
        kw = {}
        if eq is not None:
            er, ec, ev, eb = eq
            kw["A_eq"] = sparse.csr_matrix((ev, (er, ec)),
                                           shape=(max(er) + 1, self.nv))
            kw["b_eq"] = np.asarray(eb, dtype=float)
        res = linprog(np.array(self.c), A_ub=A,
                      b_ub=np.array(self.rhs, dtype=float),
                      bounds=bounds, method="highs", **kw)
        if not res.success:
            raise RuntimeError(f"crop planner solve failed: {res.message}")
        return res


def _prep(binary, img_w, img_h, p):
    """Shared preprocessing: extent tracks, phantoms, pads, floors."""
    n = binary.shape[0]
    (bx0, bx1, by0, by1), known = _extent_tracks(binary)
    loop = bool(p.get("seamless_loop")) and n > 1
    s0 = int(p.get("end_tightening_window", 0)) if not loop else 0
    if s0:
        hold = lambda a: np.concatenate(
            [np.full(s0, a[0]), a, np.full(s0, a[-1])])
        bx0, bx1, by0, by1 = hold(bx0), hold(bx1), hold(by0), hold(by1)
    m = len(bx0)
    scale = p["crop_scale"]
    padx = (scale - 1) / 2 * (bx1 - bx0 + 1)
    pady = (scale - 1) / 2 * (by1 - by0 + 1)

    floor = float(p.get("min_padding_allowed", 0.0))
    sx0, sx1, sy0, sy1 = _median_tracks([bx0, bx1, by0, by1],
                                        p.get("min_padding_allowed_window", 16))
    pad_fx = floor * (scale - 1) / 2 * (sx1 - sx0 + 1)
    pad_fy = floor * (scale - 1) / 2 * (sy1 - sy0 + 1)
    # floor datums, clamped to the image (a side pinned at the image edge
    # is exempt from its promise)
    flo_x = np.minimum(sx1 + 1 + pad_fx, img_w)
    fhi_x = np.maximum(sx0 - pad_fx, 0)
    flo_y = np.minimum(sy1 + 1 + pad_fy, img_h)
    fhi_y = np.maximum(sy0 - pad_fy, 0)

    # soft targets: the full promise, clamped to the image
    lx = np.maximum(bx0 - padx, 0)
    rx = np.minimum(bx1 + 1 + padx, img_w)
    ly = np.maximum(by0 - pady, 0)
    ry = np.minimum(by1 + 1 + pady, img_h)

    rent_scale = np.ones(m)
    if s0:
        rent_scale[:s0] = p.get("end_tightening", 0.0)
        rent_scale[m - s0:] = p.get("end_tightening", 0.0)

    return {
        "n": n, "m": m, "s0": s0, "loop": loop, "known": known,
        "bx0": bx0, "bx1": bx1, "by0": by0, "by1": by1,
        "padx": padx, "pady": pady, "floor": floor,
        "flo_x": flo_x, "fhi_x": fhi_x, "flo_y": flo_y, "fhi_y": fhi_y,
        "lx": lx, "rx": rx, "ly": ly, "ry": ry, "rent_scale": rent_scale,
    }


def _tv_rows(lp, var, tv, m, loop):
    nt = m if loop else m - 1
    for i in range(nt):
        a, b = lp.off[var] + i, lp.off[var] + (i + 1) % m
        t = lp.off[tv] + i
        lp.row([b, a, t], [1, -1, -1], 0)
        lp.row([b, a, t], [-1, 1, -1], 0)


def _solve_tracked(pre, img_w, img_h, p):
    """Scalar-size LP: W, H are single variables, shape emerges from the
    movement/rent balance. Manual aspect adds one equality."""
    m, loop = pre["m"], pre["loop"]
    nt = m if loop else m - 1
    bx0, bx1, by0, by1 = pre["bx0"], pre["bx1"], pre["by0"], pre["by1"]
    padx, pady = pre["padx"], pre["pady"]
    soft = 1.0 / max(p.get("pad_deficit_tol", 16), 1e-6)
    rent = pre["rent_scale"] / max(p.get("pad_surplus_tol", 16), 1e-6)
    mc = p.get("movement_cost", 1.0)
    eps = p.get("center_pull", 1e-4)
    extw = bx1 - bx0 + 1
    exth = by1 - by0 + 1

    lp = _LP()
    lp.var("W", 1, lo=max(extw.max(), 1), hi=img_w)
    lp.var("H", 1, lo=max(exth.max(), 1), hi=img_h)
    ub_x = np.maximum(bx0, 0)
    ub_y = np.maximum(by0, 0)
    if pre["floor"] > 0:
        ub_x = np.minimum(ub_x, pre["fhi_x"])
        ub_y = np.minimum(ub_y, pre["fhi_y"])
    lp.var("x", m, lo=0.0, hi=ub_x)
    lp.var("y", m, lo=0.0, hi=ub_y)
    lp.var("tx", nt, cost=mc)
    lp.var("ty", nt, cost=mc)
    lp.var("vx", m, cost=soft)
    lp.var("vy", m, cost=soft)
    lp.var("rx", m, cost=rent)
    lp.var("ry", m, cost=rent)
    lp.var("ux", m, cost=eps)
    lp.var("uy", m, cost=eps)

    _tv_rows(lp, "x", "tx", m, loop)
    _tv_rows(lp, "y", "ty", m, loop)
    W, H = lp.off["W"], lp.off["H"]
    for i in range(m):
        x, y = lp.off["x"] + i, lp.off["y"] + i
        vx, vy = lp.off["vx"] + i, lp.off["vy"] + i
        rx, ry = lp.off["rx"] + i, lp.off["ry"] + i
        ux, uy = lp.off["ux"] + i, lp.off["uy"] + i
        lp.row([x, W], [-1, -1], -(bx1[i] + 1))         # containment right
        lp.row([x, W], [1, 1], img_w)                   # image right
        lp.row([y, H], [-1, -1], -(by1[i] + 1))
        lp.row([y, H], [1, 1], img_h)
        lp.row([x, vx], [1, -1], pre["lx"][i])          # soft pad, left
        lp.row([x, W, vx], [-1, -1, -1], -pre["rx"][i])  # soft pad, right
        lp.row([y, vy], [1, -1], pre["ly"][i])
        lp.row([y, H, vy], [-1, -1, -1], -pre["ry"][i])
        lp.row([W, rx], [1, -1], min(extw[i] + 2 * padx[i], img_w))
        lp.row([H, ry], [1, -1], min(exth[i] + 2 * pady[i], img_h))
        if pre["floor"] > 0:
            lp.row([x, W], [-1, -1], -pre["flo_x"][i])
            lp.row([y, H], [-1, -1], -pre["flo_y"][i])
        cx = (bx0[i] + bx1[i] + 1) / 2
        cy = (by0[i] + by1[i] + 1) / 2
        lp.row([x, W, ux], [1, 0.5, -1], cx)
        lp.row([x, W, ux], [-1, -0.5, -1], -cx)
        lp.row([y, H, uy], [1, 0.5, -1], cy)
        lp.row([y, H, uy], [-1, -0.5, -1], -cy)

    eq = None
    ar = p.get("aspect_ratio", 0.0)
    if ar > 0:
        eq = ([0, 0], [W, H], [1.0, -ar], [0.0])
    res = lp.solve(eq=eq)
    o = lp.off
    return (res.x[W], res.x[H],
            res.x[o["x"]:o["x"] + m], res.x[o["y"]:o["y"] + m])


def _yielded_floors(pre, ar, img_w, img_h):
    """Relax the floor datums per side wherever containment, the image, or
    the aspect lock leaves no room."""
    bx0, bx1, by0, by1 = pre["bx0"], pre["bx1"], pre["by0"], pre["by1"]
    wmax = min(img_w, ar * img_h)
    hmax = min(img_h, img_w / ar)
    flo_x, fhi_x = pre["flo_x"].copy(), pre["fhi_x"].copy()
    flo_y, fhi_y = pre["flo_y"].copy(), pre["fhi_y"].copy()
    ex = np.maximum(flo_x - fhi_x - wmax, 0)
    flo_x = np.maximum(flo_x - ex / 2, bx1 + 1)
    fhi_x = np.minimum(fhi_x + ex / 2, np.maximum(bx0, 0))
    flo_x = np.minimum(flo_x, np.maximum(bx0, 0) + wmax)
    fhi_x = np.maximum(fhi_x, bx1 + 1 - wmax)
    ey = np.maximum(flo_y - fhi_y - hmax, 0)
    flo_y = np.maximum(flo_y - ey / 2, by1 + 1)
    fhi_y = np.minimum(fhi_y + ey / 2, np.maximum(by0, 0))
    flo_y = np.minimum(flo_y, np.maximum(by0, 0) + hmax)
    fhi_y = np.maximum(fhi_y, by1 + 1 - hmax)
    return flo_x, fhi_x, flo_y, fhi_y


def _solve_zoomed_at(pre, ar, img_w, img_h, p):
    """Inner LP for one aspect candidate. Width is ar*h throughout; rent
    is charged on h beyond the tightest aspect-respecting padded need, so
    width the ratio forces is never billed."""
    m, loop = pre["m"], pre["loop"]
    nt = m if loop else m - 1
    bx0, bx1, by0, by1 = pre["bx0"], pre["bx1"], pre["by0"], pre["by1"]
    padx, pady = pre["padx"], pre["pady"]
    extw = bx1 - bx0 + 1
    exth = by1 - by0 + 1
    soft = 1.0 / max(p.get("pad_deficit_tol", 16), 1e-6)
    # need is aspect-aware (width the ratio forces is never billed), but
    # the excess is billed at its (1+ar) box magnitude so wide and narrow
    # candidates pay the same for the same true slack
    rent = (1.0 + ar) * pre["rent_scale"] / max(p.get("pad_surplus_tol", 16), 1e-6)
    mc = p.get("movement_cost", 1.0)
    sc = p.get("resize_cost", 2.0) * (1.0 + ar)
    eps = p.get("center_pull", 1e-4)

    hmax = min(img_h, img_w / ar)
    hmin = np.maximum(np.maximum(exth, extw / ar), 1.0)
    if hmin.max() > hmax + 1e-6:
        return None  # this aspect cannot contain the subject in-image
    need_h = np.minimum(
        np.maximum(exth + 2 * pady, (extw + 2 * padx) / ar), hmax)

    floors = pre["floor"] > 0
    if floors:
        flo_x, fhi_x, flo_y, fhi_y = _yielded_floors(pre, ar, img_w, img_h)

    lp = _LP()
    ub_x = np.maximum(bx0, 0)
    ub_y = np.maximum(by0, 0)
    if floors:
        ub_x = np.minimum(ub_x, fhi_x)
        ub_y = np.minimum(ub_y, fhi_y)
    lp.var("x", m, lo=0.0, hi=ub_x)
    lp.var("y", m, lo=0.0, hi=ub_y)
    lp.var("h", m, lo=hmin, hi=hmax)
    lp.var("tx", nt, cost=mc)
    lp.var("ty", nt, cost=mc)
    lp.var("th", nt, cost=sc)
    lp.var("vx", m, cost=soft)
    lp.var("vy", m, cost=soft)
    lp.var("r", m, cost=rent)
    lp.var("ux", m, cost=eps)
    lp.var("uy", m, cost=eps)

    _tv_rows(lp, "x", "tx", m, loop)
    _tv_rows(lp, "y", "ty", m, loop)
    _tv_rows(lp, "h", "th", m, loop)

    rate = p.get("max_zoom_rate", 0.0)
    if rate and rate > 1.0:
        for i in range(nt):
            a, b = lp.off["h"] + i, lp.off["h"] + (i + 1) % m
            lp.row([b, a], [1, -rate], 0)
            lp.row([a, b], [1, -rate], 0)

    for i in range(m):
        x, y, h = lp.off["x"] + i, lp.off["y"] + i, lp.off["h"] + i
        vx, vy = lp.off["vx"] + i, lp.off["vy"] + i
        r = lp.off["r"] + i
        ux, uy = lp.off["ux"] + i, lp.off["uy"] + i
        lp.row([x, h], [-1, -ar], -(bx1[i] + 1))        # containment right
        lp.row([x, h], [1, ar], img_w)                  # image right
        lp.row([y, h], [-1, -1], -(by1[i] + 1))
        lp.row([y, h], [1, 1], img_h)
        lp.row([x, vx], [1, -1], pre["lx"][i])
        lp.row([x, h, vx], [-1, -ar, -1], -pre["rx"][i])
        lp.row([y, vy], [1, -1], pre["ly"][i])
        lp.row([y, h, vy], [-1, -1, -1], -pre["ry"][i])
        lp.row([h, r], [1, -1], need_h[i])
        if floors:
            lp.row([x, h], [-1, -ar], -flo_x[i])
            lp.row([y, h], [-1, -1], -flo_y[i])
        cx = (bx0[i] + bx1[i] + 1) / 2
        cy = (by0[i] + by1[i] + 1) / 2
        lp.row([x, h, ux], [1, 0.5 * ar, -1], cx)
        lp.row([x, h, ux], [-1, -0.5 * ar, -1], -cx)
        lp.row([y, h, uy], [1, 0.5, -1], cy)
        lp.row([y, h, uy], [-1, -0.5, -1], -cy)

    res = lp.solve()
    o = lp.off
    # the objective is the sweep score: floors carry the padding tier as
    # hard rows, so candidates compare on how cheaply they serve the
    # cell's own priced trade above them
    viol = res.x[o["vx"]:o["vx"] + m].sum() + res.x[o["vy"]:o["vy"] + m].sum()
    h_t = res.x[o["h"]:o["h"] + m]
    # dead width -- width this ratio forces beyond the frame's padded
    # horizontal need -- is billed in the score only, at the same rent as
    # height slack. Inside the LP that width stays free (billing it there
    # collapses manual wide aspects), but across candidates it
    # must cost, or the sweep drifts to the widest feasible shape whenever
    # the vertical need pins the height. Capped at need_h: excess above
    # need_h already carries its width via the (1+ar) rent.
    w_need = np.minimum(extw + 2 * padx, img_w)
    dead_w = np.maximum(ar * np.minimum(h_t, need_h) - w_need, 0)
    dead = ((pre["rent_scale"] * dead_w).sum()
            / max(p.get("pad_surplus_tol", 16), 1e-6))
    return {
        "ar": ar, "score": res.fun + dead, "viol": viol,
        "x": res.x[o["x"]:o["x"] + m], "y": res.x[o["y"]:o["y"] + m],
        "h": h_t,
    }


def _aspect_candidates(pre, img_w, img_h, n_known):
    """Hull of the per-frame shape quantiles and the padded union aspect:
    the winner lives between what the subject looks like and where it
    goes."""
    bx0, bx1, by0, by1 = pre["bx0"], pre["bx1"], pre["by0"], pre["by1"]
    padx, pady = pre["padx"], pre["pady"]
    frame_ar = (bx1 - bx0 + 1) / (by1 - by0 + 1)
    qs = np.quantile(frame_ar, [0.05, 0.25, 0.5, 0.75, 0.95])
    ux0 = (bx0 - padx).min()
    ux1 = (bx1 + 1 + padx).max()
    uy0 = (by0 - pady).min()
    uy1 = (by1 + 1 + pady).max()
    union_ar = max(ux1 - ux0, 1.0) / max(uy1 - uy0, 1.0)
    lo = min(qs.min(), union_ar)
    hi = max(qs.max(), union_ar)
    grid = np.geomspace(lo, hi, 7) if hi > lo * 1.02 else np.array([lo])
    cands = np.concatenate([grid, qs, [union_ar]])
    # feasibility band: the box must contain every frame inside the image
    extw = bx1 - bx0 + 1
    exth = by1 - by0 + 1
    band_lo = extw.max() / img_h
    band_hi = img_w / exth.max()
    cands = np.clip(cands, band_lo, band_hi)
    cands = np.unique(np.round(cands, 3))
    keep = [cands[0]]
    for a in cands[1:]:
        if a > keep[-1] * 1.02:
            keep.append(a)
    return keep


def _quantize_steps(v, step, upper):
    base = v.min()
    k = np.ceil(np.log(v / base) / np.log(step) - 1e-9)
    return np.minimum(base * step ** k, upper)


def plan(binary, sel, p, divisible_by=16):
    """binary: (n, H, W) bool mask stack. Returns (boxes, info)."""
    n, img_h, img_w = binary.shape
    if not binary.any():
        raise ValueError("all masks are empty, nothing to plan")
    scale = p["crop_scale"]

    if sel == "combined":
        ys, xs = np.nonzero(binary.any(axis=0))
        w0 = (xs.max() - xs.min() + 1) * scale
        h0 = (ys.max() - ys.min() + 1) * scale
        ar = p.get("aspect_ratio", 0.0)
        if ar > 0:
            if w0 < h0 * ar:
                w0 = h0 * ar
            else:
                h0 = w0 / ar
        w = min(math.ceil(w0 / divisible_by) * divisible_by,
                (img_w // divisible_by) * divisible_by or img_w)
        h = min(math.ceil(h0 / divisible_by) * divisible_by,
                (img_h // divisible_by) * divisible_by or img_h)
        x = min(max(0, round((xs.min() + xs.max() + 1) / 2 - w / 2)), img_w - w)
        y = min(max(0, round((ys.min() + ys.max() + 1) / 2 - h / 2)), img_h - h)
        box = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        return [box] * n, {"aspect": w / h}

    pre = _prep(binary, img_w, img_h, p)
    m, s0 = pre["m"], pre["s0"]
    bx0, bx1, by0, by1 = pre["bx0"], pre["bx1"], pre["by0"], pre["by1"]

    if sel == "tracked":
        Wf, Hf, xt, yt = _solve_tracked(pre, img_w, img_h, p)
        w = min(math.ceil(Wf / divisible_by) * divisible_by,
                (img_w // divisible_by) * divisible_by or img_w)
        h = min(math.ceil(Hf / divisible_by) * divisible_by,
                (img_h // divisible_by) * divisible_by or img_h)
        lo_x = np.maximum(bx1 + 1 - w, 0)
        hi_x = np.minimum(np.maximum(bx0, 0), img_w - w)
        lo_y = np.maximum(by1 + 1 - h, 0)
        hi_y = np.minimum(np.maximum(by0, 0), img_h - h)
        if pre["floor"] > 0:
            lo_x = np.maximum(lo_x, np.minimum(pre["flo_x"] - w, hi_x))
            hi_x = np.minimum(hi_x, np.maximum(pre["fhi_x"], lo_x))
            lo_y = np.maximum(lo_y, np.minimum(pre["flo_y"] - h, hi_y))
            hi_y = np.minimum(hi_y, np.maximum(pre["fhi_y"], lo_y))
        xi = _round_corridor(xt, lo_x, hi_x)[s0:s0 + n]
        yi = _round_corridor(yt, lo_y, hi_y)[s0:s0 + n]
        boxes = [{"x": int(xi[i]), "y": int(yi[i]),
                  "width": int(w), "height": int(h)} for i in range(n)]
        return boxes, {"aspect": w / h, "W": Wf, "H": Hf}

    # zoomed
    ar_manual = p.get("aspect_ratio", 0.0)
    if ar_manual > 0:
        extw = bx1 - bx0 + 1
        exth = by1 - by0 + 1
        ar = min(max(ar_manual, extw.max() / img_h), img_w / exth.max())
        best = _solve_zoomed_at(pre, ar, img_w, img_h, p)
        if best is None:
            raise RuntimeError("requested aspect_ratio cannot contain the subject")
        tried = [best]
    else:
        tried = []
        for ar in _aspect_candidates(pre, img_w, img_h, pre["known"].sum()):
            sol = _solve_zoomed_at(pre, ar, img_w, img_h, p)
            if sol is not None:
                tried.append(sol)
        if not tried:
            raise RuntimeError("no feasible aspect candidate")
        best = min(tried, key=lambda s: s["score"])
        for ar in (best["ar"] * 0.93, best["ar"] * 1.075):
            sol = _solve_zoomed_at(pre, ar, img_w, img_h, p)
            if sol is not None and sol["score"] < best["score"]:
                best = sol

    ar = best["ar"]
    h = best["h"]
    if p.get("zoom_step", 1.0) > 1.0:
        h = _quantize_steps(h, p["zoom_step"], min(img_h, img_w / ar))
    w = np.minimum(h * ar, img_w)
    bw = np.minimum(np.ceil(w - 1e-6), img_w).astype(int)
    bh = np.minimum(np.ceil(h - 1e-6), img_h).astype(int)
    lo_x = np.maximum(bx1 + 1 - bw, 0)
    hi_x = np.minimum(np.maximum(bx0, 0), img_w - bw)
    lo_y = np.maximum(by1 + 1 - bh, 0)
    hi_y = np.minimum(np.maximum(by0, 0), img_h - bh)
    if pre["floor"] > 0:
        flo_x, fhi_x, flo_y, fhi_y = _yielded_floors(pre, ar, img_w, img_h)
        lo_x = np.maximum(lo_x, np.minimum(flo_x - bw, hi_x))
        hi_x = np.minimum(hi_x, np.maximum(fhi_x, lo_x))
        lo_y = np.maximum(lo_y, np.minimum(flo_y - bh, hi_y))
        hi_y = np.minimum(hi_y, np.maximum(fhi_y, lo_y))
    xi = _round_corridor(best["x"], lo_x, hi_x)[s0:s0 + n]
    yi = _round_corridor(best["y"], lo_y, hi_y)[s0:s0 + n]
    bw, bh = bw[s0:s0 + n], bh[s0:s0 + n]
    boxes = [{"x": int(min(max(xi[i], 0), img_w - bw[i])),
              "y": int(min(max(yi[i], 0), img_h - bh[i])),
              "width": int(bw[i]), "height": int(bh[i])} for i in range(n)]
    return boxes, {"aspect": ar, "swept": ar_manual <= 0,
                   "candidates": len(tried)}
