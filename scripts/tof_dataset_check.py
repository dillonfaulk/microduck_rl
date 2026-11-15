"""Validate the sim ToF model against Pollen's real recordings (rung 6 Stage 2, item 2.1).

    uv run python scripts/tof_dataset_check.py ~/microduck_refs/slam_gt [--out DIR]

Dataset: `pollen-robotics/SLAM_JJ_Salon_with_GT` (Apache-2.0), the subset with no films:
`sessions/<s>/session/*/{state,tof,meta}`, `sessions/<s>/ground_truth/`, `room/`.
It is read as data only; nothing from its `code/` directory is imported or run.

Part A reproduces Pollen's own sensor-model check (their `maploc_truth/scripts/tof_check.py`,
reimplemented here). The real returns are placed with the TRUE pose, and the script measures how
often they land within 5 cm of an occupied cell of the scanned room. It does this once per
sensor-model variant (range convention, latency, pitch offset). Pollen's published figures for s2
are 58 % (as first configured) -> 69 % (axial) -> 81 % (axial + 60 ms + 3 deg)
(microduck maploc/src/project.rs:62-72).

Part B runs the forward direction, the one a simulator uses. It ray-casts the sim sensor model
into the scanned room (point cloud voxelised at 2 cm, plus the z = 0 floor) from the true pose,
then compares the predicted range with each zone's measured range. From the same casts it
measures the real sensor's dropout rate (status not in {5, 9}) by grid row and by predicted
range: the numbers the sim's dropout model uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VALID = (5, 9)
VOXEL = 0.02


# --- small geometry helpers (scalar-first quaternions, the robot's wire convention) ---------------

def quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def se2_compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1],
                     a[..., 1] + s * b[..., 0] + c * b[..., 1], wrap(a[..., 2] + b[..., 2])], axis=-1)


def se2_inv(a):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-(c * a[..., 0] + s * a[..., 1]), -(-s * a[..., 0] + c * a[..., 1]), wrap(-a[..., 2])], axis=-1)


# --- loading -----------------------------------------------------------------------------------

def load_session(root: Path, name: str):
    sess = next((root / "sessions" / name / "session").iterdir())
    meta = json.loads((sess / "meta.json").read_text())
    beams = np.array(meta["model"]["tof_beams"])                       # (64, 3) sensor frame
    S = {"t": [], "pos": [], "R": [], "tofp": [], "tofR": [], "ok": []}
    for line in open(sess / "state.jsonl"):
        p = json.loads(line)["params"]
        if not p.get("t_ns") or not p.get("frames"):
            continue
        S["t"].append(p["t_ns"] * 1e-9)
        S["pos"].append(p["odom"]["position"])
        S["R"].append(quat_to_R(p["imu"]["quat"]))
        f = p["frames"]["tof"]
        S["tofp"].append(f["pos"])
        S["tofR"].append(quat_to_R(f["quat"]))
        s = p.get("safety") or {}
        S["ok"].append(p.get("policy") == "walk" and not s.get("picked_up") and not s.get("fallen"))
    S = {k: np.array(v) for k, v in S.items()}
    odom2d = np.column_stack([S["pos"][:, 0], S["pos"][:, 1], np.arctan2(S["R"][:, 1, 0], S["R"][:, 0, 0])])
    gt = np.loadtxt(root / "sessions" / name / "ground_truth" / "ground_truth.csv", delimiter=",", skiprows=1)
    Rg = np.array([quat_to_R(q) for q in gt[:, 4:8]])
    truth2d = np.column_stack([gt[:, 1], gt[:, 2], np.arctan2(Rg[:, 1, 0], Rg[:, 0, 0])])
    tg = gt[:, 0] * 1e-9
    idx = np.clip(np.searchsorted(S["t"], tg), 1, len(S["t"]) - 1)
    T_wo = se2_compose(truth2d, se2_inv(odom2d[idx]))                 # world <- odom, per truth sample
    frames = [json.loads(line)["params"] for line in open(sess / "tof.jsonl")]
    frames = [f for f in frames if f.get("t_ns")]
    return {"S": S, "tg": tg, "T_wo": T_wo, "frames": frames, "beams": beams}


def load_room(root: Path):
    z = np.load(root / "room" / "truth.npz")
    cells, res, origin = z["cells"], float(z["res"]), np.array(z["origin"], float)
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(cells != 1) * res
    except ImportError:  # pragma: no cover
        sys.exit("scipy is needed for the distance field")
    return {"cells": cells, "res": res, "origin": origin, "dist": dist}


def load_voxels(root: Path):
    """room.ply (binary little-endian, float xyz + uchar rgb) -> boolean voxel grid, floor excluded."""
    raw = (root / "room" / "room.ply").read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    n = int([ln for ln in head.decode().splitlines() if ln.startswith("element vertex")][0].split()[-1])
    rec = np.frombuffer(body[: n * 15], dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "u1", 3)]))
    pts = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(float)
    pts = pts[pts[:, 2] > 0.03]          # the scan's floor reads ~25 mm thick; the floor is the z = 0 plane
    lo = pts.min(0) - VOXEL
    ijk = np.floor((pts - lo) / VOXEL).astype(int)
    occ = np.zeros(ijk.max(0) + 2, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return {"occ": occ, "lo": lo, "n_points": int(n)}


# --- pose at a frame ----------------------------------------------------------------------------

def frame_pose(sess, t, pitch_off):
    """(origin_w (3,), R_sensor_w (3,3)) at capture time t, or None. Pollen's pipeline: odometry
    interpolated at t, carried into the world by the truth-derived world<-odom offset."""
    S, tg = sess["S"], sess["tg"]
    k = np.searchsorted(S["t"], t)
    if k <= 0 or k >= len(S["t"]) or not S["ok"][k]:
        return None
    g = min(max(np.searchsorted(tg, t), 0), len(tg) - 1)
    if abs(tg[g] - t) > 0.15:
        return None
    u = (t - S["t"][k - 1]) / (S["t"][k] - S["t"][k - 1])
    pos = S["pos"][k - 1] + u * (S["pos"][k] - S["pos"][k - 1])
    R = S["R"][k]
    o = pos + R @ S["tofp"][k]
    Rs = R @ S["tofR"][k] @ rot_y(pitch_off)
    T = sess["T_wo"][g]
    c, s = np.cos(T[2]), np.sin(T[2])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ o + np.array([T[0], T[1], 0.0]), Rz @ Rs


# --- Part A: Pollen's metric, reimplemented ------------------------------------------------------

def part_a(sess, room, axial, latency_s, pitch_off):
    beams, d_all = sess["beams"], []
    for f in sess["frames"]:
        pose = frame_pose(sess, f["t_ns"] * 1e-9 - latency_s, pitch_off)
        if pose is None:
            continue
        o, Rs = pose
        d = np.array(f["distance_mm"], float) / 1000.0
        st = np.array(f["status"])
        r = d / beams[:, 0] if axial else d
        p = o + r[:, None] * (beams @ Rs.T)
        h = np.hypot(p[:, 0] - o[0], p[:, 1] - o[1])
        keep = (d > 0.08) & (r < 2.0) & (p[:, 2] > 0.03 + 0.03 * h) & (p[:, 2] < 0.40) & np.isin(st, VALID)
        if not keep.any():
            continue
        ij = np.floor((p[keep, :2] - room["origin"]) / room["res"]).astype(int)
        hh, ww = room["cells"].shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < ww) & (ij[:, 1] >= 0) & (ij[:, 1] < hh)
        dd = np.full(len(ij), np.inf)
        dd[inside] = room["dist"][ij[inside, 1], ij[inside, 0]]
        d_all.append(dd)
    d = np.concatenate(d_all)
    return {"n": int(len(d)), "within_5cm": float(np.mean(d <= 0.05)), "within_10cm": float(np.mean(d <= 0.10)),
            "median_cm": float(np.median(d) * 100)}


# --- Part B: the forward model ------------------------------------------------------------------

def cast(vox, origins, dirs, max_r=4.0, step=0.01):
    """Distance along each unit ray to the first occupied voxel or the z = 0 floor (inf if none)."""
    ts = np.arange(step, max_r + step, step)
    out = np.full(len(dirs), np.inf)
    floor_t = np.where(dirs[:, 2] < -1e-6, -origins[:, 2] / np.minimum(dirs[:, 2], -1e-6), np.inf)
    P = origins[:, None, :] + dirs[:, None, :] * ts[None, :, None]          # (n, steps, 3)
    ijk = np.floor((P - vox["lo"]) / VOXEL).astype(int)
    shp = np.array(vox["occ"].shape)
    ok = np.all((ijk >= 0) & (ijk < shp), axis=-1)
    hit = np.zeros(ok.shape, dtype=bool)
    hit[ok] = vox["occ"][ijk[ok][:, 0], ijk[ok][:, 1], ijk[ok][:, 2]]
    first = np.where(hit.any(1), hit.argmax(1), -1)
    out[first >= 0] = ts[first[first >= 0]]
    return np.minimum(out, floor_t)


def part_b(sess, vox, axial, latency_s, pitch_off, collect_status=False):
    beams = sess["beams"]
    meas, pred, rows, cols, stat = [], [], [], [], []
    for f in sess["frames"]:
        pose = frame_pose(sess, f["t_ns"] * 1e-9 - latency_s, pitch_off)
        if pose is None:
            continue
        o, Rs = pose
        dirs = beams @ Rs.T
        t = cast(vox, np.repeat(o[None], 64, 0), dirs)
        pred.append(t * beams[:, 0] if axial else t)                        # what the sensor would report
        meas.append(np.array(f["distance_mm"], float) / 1000.0)
        stat.append(np.array(f["status"]))
        rows.append(np.arange(64) // 8)
        cols.append(np.arange(64) % 8)
    m, p, r, c, s = (np.concatenate(x) for x in (meas, pred, rows, cols, stat))
    v = np.isin(s, VALID) & np.isfinite(p) & (m > 0.08)
    err = m[v] - p[v]
    corner = np.isin(c[v], (0, 7)) & np.isin(r[v], (0, 7))
    res = {"n_valid": int(v.sum()), "median_err_mm": float(np.median(err) * 1000),
           "within_5cm": float(np.mean(np.abs(err) <= 0.05)), "within_10cm": float(np.mean(np.abs(err) <= 0.10)),
           "corner_median_err_mm": float(np.median(err[corner]) * 1000) if corner.any() else None,
           "centre_median_err_mm": float(np.median(err[~corner]) * 1000) if (~corner).any() else None}
    if collect_status:
        res["status_counts"] = {int(k): int(n) for k, n in zip(*np.unique(s, return_counts=True))}
        res["valid_by_row"] = [float(np.mean(np.isin(s[r == i], VALID))) for i in range(8)]
        bins = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf)
        res["range_bins_m"] = [b if np.isfinite(b) else "none" for b in bins[:-1]]
        table = []
        for i in range(8):
            line = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (r == i) & (p >= lo) & (p < hi) if np.isfinite(hi) else (r == i) & ~np.isfinite(p)
                line.append(None if sel.sum() < 50 else round(float(np.mean(np.isin(s[sel], VALID))), 3))
            table.append(line)
        res["valid_by_row_and_pred_range"] = table
        allb = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p >= lo) & (p < hi) if np.isfinite(hi) else ~np.isfinite(p)
            allb.append(None if sel.sum() < 50 else round(float(np.mean(np.isin(s[sel], VALID))), 3))
        res["valid_by_pred_range"] = allb
        # range noise vs range, on the best model's valid zones
        nb = []
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 4.0)):
            sel = (p[v] >= lo) & (p[v] < hi)
            e = err[sel]
            e = e[np.abs(e) < 0.15]                  # inliers: the spread, not the map's errors
            nb.append({"range_m": [lo, hi], "n": int(sel.sum()),
                       "inlier_std_mm": round(float(np.std(e) * 1000), 1) if len(e) > 50 else None})
        res["noise_by_range"] = nb
    return res


VARIANTS = (
    ("as first configured (slant, 0 ms, 0 deg)", False, 0.0, 0.0),
    ("axial", True, 0.0, 0.0),
    ("axial + 60 ms", True, 0.06, 0.0),
    ("axial + 60 ms + 3 deg down (upstream defaults)", True, 0.06, np.radians(3)),
    ("slant + 60 ms + 3 deg (axial off)", False, 0.06, np.radians(3)),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sessions", default="s2,s3")
    args = ap.parse_args()
    room, vox = load_room(args.root), load_voxels(args.root)
    print(f"room: grid {room['cells'].shape} @ {room['res']} m; cloud {vox['n_points']} pts -> voxels {vox['occ'].shape} @ {VOXEL} m")
    report = {"dataset": str(args.root), "sessions": {}}
    for name in args.sessions.split(","):
        sess = load_session(args.root, name)
        print(f"\n== {name}: {len(sess['frames'])} ToF frames, {len(sess['S']['t'])} state samples, {len(sess['tg'])} truth poses")
        out = {"part_a": {}, "part_b": {}}
        print("  Part A (Pollen's metric: real returns, true pose, land within 5 cm of an obstacle cell)")
        for label, axial, lat, po in VARIANTS:
            a = part_a(sess, room, axial, lat, po)
            out["part_a"][label] = a
            print(f"    {label:48s} n={a['n']:6d}  <=5cm {a['within_5cm']:5.1%}  <=10cm {a['within_10cm']:5.1%}  median {a['median_cm']:.1f} cm")
        print("  Part B (sim model ray-cast into the scanned room vs the measured range, status 5/9 zones)")
        for i, (label, axial, lat, po) in enumerate(VARIANTS):
            b = part_b(sess, vox, axial, lat, po, collect_status=(i == 3))
            out["part_b"][label] = b
            print(f"    {label:48s} n={b['n_valid']:6d}  |err|<=5cm {b['within_5cm']:5.1%}  <=10cm {b['within_10cm']:5.1%}  "
                  f"median err {b['median_err_mm']:+6.1f} mm  corner {b['corner_median_err_mm']:+6.1f}  centre {b['centre_median_err_mm']:+6.1f}")
        st = out["part_b"][VARIANTS[3][0]]
        print(f"  status counts: {st['status_counts']}")
        print("  valid (5/9) fraction by row (row 0 = top): " + " ".join(f"{x:.2f}" for x in st["valid_by_row"]))
        print("  valid fraction by predicted range " + str(st["range_bins_m"]) + ": " + str(st["valid_by_pred_range"]))
        for i, line in enumerate(st["valid_by_row_and_pred_range"]):
            print(f"    row {i}: {line}")
        print("  inlier range noise: " + ", ".join(f"{d['range_m']}: {d['inlier_std_mm']} mm (n={d['n']})" for d in st["noise_by_range"]))
        report["sessions"][name] = out
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tof_dataset_check.json").write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out / 'tof_dataset_check.json'}")


if __name__ == "__main__":
    main()
