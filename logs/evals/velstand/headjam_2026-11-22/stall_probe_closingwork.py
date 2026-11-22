import sys, math, numpy as np, mujoco
sys.path.insert(0, "scripts")
from kick_battery import SCENES, KickSim
from kick_mission import setup_goal, setup_field
V = "logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx"
S = np.load(sys.argv[1])
sim = KickSim(V, SCENES["field"]); setup_goal(sim); setup_field(sim)
m, d, p = sim.model, sim.data, sim.policy
def load():
    sim.reset(0, 0, 0)
    d.qpos[:] = S["qpos"]; d.qvel[:] = S["qvel"]
    p.last_action[:] = S["last_action"]
    if p.action_buffer is not None:
        for a, b in zip(p.action_buffer, S["buf"]): a[:] = b
    p.head_offset[:] = S["head"]; sim.set_twist(tuple(S["vel_cmd"][:3]))
    mujoco.mj_forward(m, d)
load()
(x, y, yaw), tilt, z = sim.pose()
w, qx, qy, qz = d.qpos[sim.qa + 3:sim.qa + 7]
roll = math.degrees(math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * qy - qz * qx)))))
print(f"t {float(S['t']):.1f}: tilt {tilt:.1f} deg (roll {roll:+.1f}, pitch {pitch:+.1f}), trunk z {z:.3f} m, head {S['head']}, cmd {S['vel_cmd']}")
names = [m.joint(m.dof_jntid[m.jnt_dofadr[j]]).name for j in range(m.njnt)]
q = np.array([d.qpos[i] for i in p.joint_qpos_indices]) - p.default_pose
jn = [m.joint(j).name for j in range(m.njnt) if m.jnt_qposadr[j] in list(p.joint_qpos_indices)]
print("joint offsets from HOME (deg):", {n: round(math.degrees(v), 1) for n, v in zip(jn, q)})
print("last action (rad):", np.round(S["last_action"], 2))
cons = set()
for i in range(d.ncon):
    c = d.contact[i]
    a, b = m.geom(c.geom1), m.geom(c.geom2)
    ba, bb = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
    cons.add(tuple(sorted((ba, bb))))
print("contacts (body pairs):", sorted(cons))
print("projected gravity (trunk):", np.round(d.xmat[sim.trunk_body].reshape(3, 3).T @ np.array([0, 0, -1.0]), 3))
# does it stay stuck in isolation, with zero command and head neutral?
for label, setup in (("as dumped", lambda: None), ("head neutral", lambda: p.head_offset.__setitem__(slice(None), 0.0))):
    load(); setup()
    tl = []
    for k in range(1000):
        sim.step(); tl.append(sim.pose()[1])
    print(f"{label}: tilt after 5/10/20 s {tl[249]:.1f} / {tl[499]:.1f} / {tl[999]:.1f}; min over 20 s {min(tl):.1f}")
# perturbations: small pushes, a walk command
for label, fn in (("push 0.3 m/s fwd", lambda: sim.push(0.3*math.cos(yaw), 0.3*math.sin(yaw))),
                  ("push 0.3 m/s back", lambda: sim.push(-0.3*math.cos(yaw), -0.3*math.sin(yaw))),
                  ("push 0.3 m/s left", lambda: sim.push(-0.3*math.sin(yaw), 0.3*math.cos(yaw))),
                  ("push 0.3 m/s right", lambda: sim.push(0.3*math.sin(yaw), -0.3*math.cos(yaw)))):
    load(); fn(); tl = []
    for k in range(500):
        sim.step(); tl.append(sim.pose()[1])
    print(f"{label}: tilt after 10 s {tl[-1]:.1f}, min {min(tl):.1f}")
for cmd in ((0.3, 0, 0), (-0.2, 0, 0), (0, 0, 0.8)):
    load(); sim.set_twist(cmd); tl = []
    for k in range(500):
        sim.step(); tl.append(sim.pose()[1])
    print(f"command {cmd}: tilt after 10 s {tl[-1]:.1f}, min {min(tl):.1f}")
