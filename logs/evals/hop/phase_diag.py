# OmniHop phase diagnostic (2026-11-25): usage python logs/evals/hop/phase_diag.py <onnx> <request delay s>
# Where does the displacement come from? Per direction: displacement during the longest flight (lift-off ->
# touchdown), at the end of that touchdown's landing, and at 2.5 s; single-foot (stepping) ticks after landing.
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
    t = TG[d]; c, s = math.cos(yaw0), math.sin(yaw0)
    goal = (x0 + c * t[0] - s * t[1], y0 + s * t[0] + c * t[1])
    sim.policy.ort_session = hs.hop_session; sim.t = 0.0
    air = best = 0; lift = None; flight_disp = (0.0, 0.0); land_t = None; steps_after = 0; pre_steps = 0
    rel = lambda: hb.to_frame(sim.pose()[0][0] - x0, sim.pose()[0][1] - y0, yaw0)
    while sim.t < 2.5:
        (x, y, yaw), _, _ = sim.pose()
        tx, ty = hb.to_frame(goal[0] - x, goal[1] - y, yaw)
        hs.set_cmd((tx, ty), dl <= sim.t < dl + 0.6)
        p_before = rel()
        sim.step()
        ld, rd, _ = hs.contacts()
        if not (ld or rd):
            if air == 0: lift = p_before
            air += 1
        else:
            if air > 0 and air >= best:
                best = air; p = rel(); flight_disp = (p[0] - lift[0], p[1] - lift[1]); land_t = sim.t; pre = lift
            air = 0
            if ld != rd:
                if land_t is not None and sim.t > land_t: steps_after += 1
                elif land_t is None: pre_steps += 1
    end = rel()
    pre = pre if land_t is not None else (0.0, 0.0)
    return d, 1000 * pre[0], 1000 * pre[1], 1000 * flight_disp[0], 1000 * flight_disp[1], 1000 * end[0], 1000 * end[1], steps_after, pre_steps
if __name__ == "__main__":
    onnx, dl = sys.argv[1], float(sys.argv[2])
    with Pool(12) as p: res = p.map(run, [(onnx, d, dl, ep) for d in TG for ep in range(6)])
    print(f"{onnx.split('/')[-1]} delay {dl}   (mm, start yaw frame; medians of 6)")
    print(f"  {'dir':9s} {'before lift-off':>16s} {'during flight':>16s} {'at 2.5 s':>16s} {'1-foot ticks pre/post':>22s}")
    for d in TG:
        a = np.array([x[1:] for x in res if x[0] == d])
        md = np.median(a, axis=0)
        print(f"  {d:9s} [{md[0]:+6.1f},{md[1]:+6.1f}] [{md[2]:+6.1f},{md[3]:+6.1f}] [{md[4]:+6.1f},{md[5]:+6.1f}] {md[7]:>10.0f} / {md[6]:<6.0f}")
