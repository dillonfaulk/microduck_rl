import re

import mujoco

from mjlab_microduck.robot.microduck_constants import get_standup_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_standup_env_cfg as su

GATED = ("height_stand", "height_stand_sharp", "upright_sharp", "standing_composite", "pose_stand_legs")
UNGATED_BOOTSTRAP = ("upright_linear", "com_upward_velocity")


def test_gate_sensor_and_terms_wired():
    assert su.ENABLE_SUPPORT_CONTACT_GATE
    cfg = su.make_microduck_standup_env_cfg()
    assert "nonfoot_ground_contact" in [s.name for s in cfg.scene.sensors]
    for name in GATED:
        term = cfg.rewards[name]
        assert term.func.__name__.endswith("_feet_only"), name
        assert term.params["sensor_name"] == "nonfoot_ground_contact"
    for name in UNGATED_BOOTSTRAP:
        assert "sensor_name" not in cfg.rewards[name].params, name


def test_nonfoot_pattern_covers_every_collision_body_except_feet():
    m = get_standup_spec().compile()
    collision_bodies = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        for g in range(m.ngeom)
        if m.geom_contype[g] or m.geom_conaffinity[g]
    }
    feet = {"ankle_left", "ankle_right"}
    assert feet <= collision_bodies
    pat = re.compile(su.NONFOOT_BODIES_PATTERN)
    matched = {b for b in collision_bodies if pat.match(b)}
    assert matched == collision_bodies - feet


def test_gate_wrappers_exist():
    for fn in ("height_target_gaussian_feet_only", "upright_gaussian_at_height_feet_only",
               "standing_composite_score_feet_only", "pose_target_match_feet_only", "feet_only_support"):
        assert callable(getattr(microduck_mdp, fn))
