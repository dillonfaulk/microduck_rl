"""SCAN pivot footprint (kick_mission SCAN_WZ = 0.6, 15 s), round 3 vs 4250: how far the trunk and feet wander
from the start point while turning in place. 6 start yaws each. Post-contact risk scales with this reach."""
import math, sys
from multiprocessing import Pool
import numpy as np, mujoco
sys.path.insert(0, "scripts")
POL = {"round3": "logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx",
       "hj4250": "logs/rsl_rl/velstand/2026-11-22_19-47-54_headjamB/2026-11-22_19-47-54_headjamB_model_4250.onnx"}
def run(job):
    pol, k = job
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        import goto_point as gp
        gp.ip.NECK_CLIP_ENABLED = pol == "hj4250"
        sim = gp.DuckSim(POL[pol])
    sim.reset(0, 0, k * math.pi / 3)
    for _ in range(50): sim.step()
    m, d = sim.model, sim.data
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]
    c0 = d.qpos[sim.qa:sim.qa + 2].copy()
    sim.set_twist((0, 0, 0.6))
    trunk, reach, yaw0 = [], [], sim.pose()[0][2]; turned = 0.0; prev = yaw0
    for _ in range(750):
        sim.step()
        trunk.append(np.linalg.norm(d.qpos[sim.qa:sim.qa + 2] - c0))
        reach.append(max(np.linalg.norm(d.site_xpos[f][:2] - c0) for f in feet))
        y = sim.pose()[0][2]; turned += (y - prev + math.pi) % (2 * math.pi) - math.pi; prev = y
    return pol, max(trunk), float(np.linalg.norm(d.qpos[sim.qa:sim.qa + 2] - c0)), max(reach), turned / 15.0
with Pool(12) as p:
    res = p.map(run, [(pol, k) for pol in POL for k in range(6)])
for pol in POL:
    r = np.array([x[1:] for x in res if x[0] == pol])
    print(f"{pol:7s} trunk max excursion med {1000*np.median(r[:,0]):.0f} mm (max {1000*r[:,0].max():.0f}), end drift med {1000*np.median(r[:,1]):.0f} mm, "
          f"foot reach max med {1000*np.median(r[:,2]):.0f} mm (max {1000*r[:,2].max():.0f}), turn rate {np.median(r[:,3]):.2f} rad/s")
