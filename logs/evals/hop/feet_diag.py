# Trunk vs FEET-MIDPOINT displacement (start yaw frame): at touchdown of the longest flight and at 2.5 s.
import sys, math, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "scripts")
import hop_battery as hb
TG = {"in_place": (0, 0), "fwd": (0.03, 0), "back": (-0.03, 0), "left": (0, 0.02), "right": (0, -0.02)}
def run(job):
    onnx, d, dl, ep = job
    g = globals()
    if g.get("_k") != onnx:
        g["_hs"] = hb.HopSim(onnx, hb.DEFAULT_VELSTAND); g["_k"] = onnx
    hs = g["_hs"]; sim = hs.sim
    rng = np.random.default_rng(ep)
    sim.reset(0, 0, rng.uniform(-3, 3)); sim.policy.ort_session = hs.prestand; sim.set_twist((0, 0, 0))
    while sim.t < 1.5 - 1e-9: sim.step()
    (x0, y0, yaw0), _, _ = sim.pose()
    feet = lambda: np.mean([sim.data.site_xpos[i][:2] for i in hs.sites], axis=0)
    f0 = feet()
    t = TG[d]; c, s = math.cos(yaw0), math.sin(yaw0)
    goal = (x0 + c * t[0] - s * t[1], y0 + s * t[0] + c * t[1])
    sim.policy.ort_session = hs.hop_session; sim.t = 0.0
    air = best = 0; td_trunk = td_feet = None
    rel = lambda p: hb.to_frame(p[0] - x0, p[1] - y0, yaw0)
    relf = lambda p: hb.to_frame(p[0] - f0[0], p[1] - f0[1], yaw0)
    while sim.t < 2.5:
        (x, y, yaw), _, _ = sim.pose()
        tx, ty = hb.to_frame(goal[0] - x, goal[1] - y, yaw)
        hs.set_cmd((tx, ty), dl <= sim.t < dl + 0.6)
        sim.step()
        ld, rd, _ = hs.contacts()
        if not (ld or rd): air += 1
        else:
            if air > 0 and air >= best:
                best = air; (x, y, _), _, _ = sim.pose(); td_trunk = rel((x, y)); td_feet = relf(feet())
            air = 0
    (x, y, _), _, _ = sim.pose()
    return d, *(td_trunk or (0, 0)), *(td_feet or (0, 0)), *rel((x, y)), *relf(feet())
if __name__ == "__main__":
    onnx, dl = sys.argv[1], float(sys.argv[2])
    with Pool(12) as p: res = p.map(run, [(onnx, d, dl, ep) for d in TG for ep in range(6)])
    print(f"{onnx.split('/')[-1]} delay {dl}  (mm fwd,side, start yaw frame; medians of 6)")
    print(f"  {'dir':9s} {'trunk @touchdown':>17s} {'feet @touchdown':>17s} {'trunk @2.5s':>17s} {'feet @2.5s':>17s}")
    for d in TG:
        a = 1000 * np.median(np.array([x[1:] for x in res if x[0] == d]), axis=0)
        print(f"  {d:9s} " + " ".join(f"[{a[i]:+6.1f},{a[i+1]:+6.1f}]" for i in (0, 2, 4, 6)))
