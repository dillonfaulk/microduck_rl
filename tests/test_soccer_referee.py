import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import soccer_referee as sr  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
GOAL_SCENE = os.path.join(REPO, "src/mjlab_microduck/robot/microduck/scene_allcollisions_ball_goal.xml")


def goal_at(cx=1.0, cy=0.0, yaw=0.0):
    # the 210 mm goal: post centres +-0.115, radius 0.010
    c, s = math.cos(yaw), math.sin(yaw)
    left = (cx - s * 0.115, cy + c * 0.115)
    right = (cx + s * 0.115, cy - c * 0.115)
    return sr.Goal.from_posts(left, right, 0.010, yaw)


def roll(ref, start, heading, speed=0.22, dt=0.02, t_max=12.0, contact_at=None):
    """Feed a straight constant-speed roll; contact_at = time window (t0, t1) of post contact."""
    x, y = start
    t = 0.0
    while t <= t_max:
        c = contact_at is not None and contact_at[0] <= t <= contact_at[1]
        r = ref.update(t, x, y, c)
        if r is not None:
            return r
        x += speed * dt * math.cos(heading)
        y += speed * dt * math.sin(heading)
        t += dt
    return None


def test_goal_geometry_from_posts():
    g = goal_at()
    assert g.opening == pytest.approx(0.210)
    assert g.half_opening == pytest.approx(0.105)
    assert g.to_frame(1.5, 0.0) == pytest.approx((0.5, 0.0))
    assert g.to_frame(1.0, 0.05)[1] == pytest.approx(0.05)   # +y = left facing in


def test_centre_kick_is_a_goal_with_time_to_line():
    r = roll(sr.GoalReferee(goal_at()), (0.5, 0.0), 0.0)
    assert r["goal_outcome"] == "goal" and r["goal_went_in"]
    assert r["goal_lateral_mm"] == pytest.approx(0.0, abs=0.5)
    assert r["goal_t_line_s"] == pytest.approx(0.5 / 0.22, abs=0.03)
    assert r["goal_miss_by_mm"] is None


def test_miss_sign_and_distance():
    # 15 deg left of the line from 0.5 m: crosses at +134 mm, 29 mm outside the left post's inner face
    left = roll(sr.GoalReferee(goal_at()), (0.5, 0.0), math.radians(15))
    assert left["goal_outcome"] == "miss"
    assert left["goal_lateral_mm"] == pytest.approx(500 * math.tan(math.radians(15)), abs=1.0)
    assert left["goal_miss_by_mm"] == pytest.approx(500 * math.tan(math.radians(15)) - 105, abs=1.0)
    right = roll(sr.GoalReferee(goal_at()), (0.5, 0.0), math.radians(-15))
    assert right["goal_outcome"] == "miss" and right["goal_lateral_mm"] < 0


def test_post_contact_is_its_own_line_even_if_it_goes_in():
    r = roll(sr.GoalReferee(goal_at()), (0.5, 0.0), 0.0, contact_at=(2.0, 2.1))
    assert r["goal_outcome"] == "post" and r["goal_went_in"] and r["goal_post_hit"]


def test_post_bounce_back_resolves_at_timeout():
    ref = sr.GoalReferee(goal_at(), timeout_s=5.0)
    r = None
    for i in range(400):   # rolls in, touches the post, rolls back out
        t = i * 0.02
        x = 0.5 + 0.22 * t if t < 2.0 else 0.94 - 0.1 * (t - 2.0)
        r = ref.update(t, x, 0.1, 1.95 < t < 2.05)
        if r is not None:
            break
    assert r["goal_outcome"] == "post" and not r["goal_went_in"] and r["goal_lateral_mm"] is None


def test_no_cross_on_timeout_and_backwards_crossing_ignored():
    r = roll(sr.GoalReferee(goal_at(), timeout_s=3.0), (0.5, 0.0), math.pi)   # rolls away from the goal
    assert r["goal_outcome"] == "no_cross"
    # a ball already behind the line rolling OUT through it is not a goal
    r = roll(sr.GoalReferee(goal_at(), timeout_s=3.0), (1.2, 0.0), math.pi)
    assert r["goal_outcome"] == "no_cross"


def test_rotated_goal_frame():
    g = goal_at(cx=0.0, cy=1.0, yaw=math.pi / 2)        # goal to the north, facing north
    r = roll(sr.GoalReferee(g), (0.0, 0.4), math.pi / 2 + math.radians(5))
    assert r["goal_outcome"] == "goal"
    assert r["goal_lateral_mm"] > 0                     # 5 deg left of the line -> + lateral


def test_result_is_frozen_after_decision():
    ref = sr.GoalReferee(goal_at())
    r = roll(ref, (0.5, 0.0), 0.0)
    assert ref.update(99.0, -5.0, 3.0, True) is r


def test_field_out_of_bounds_whole_ball_rule():
    f = sr.Field(-2.0, 2.0, -1.5, 1.5)
    assert not f.out_of_bounds(2.02, 0.0, ball_radius=0.035)   # still touching the line
    assert f.out_of_bounds(2.04, 0.0, ball_radius=0.035)
    assert f.out_of_bounds(0.0, -1.6)


def test_goal_scene_matches_the_referee_geometry():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(GOAL_SCENE)
    d = mujoco.MjData(m)
    assert m.nmocap == 2
    for name in ("goal_a", "goal_b"):
        mid = m.body_mocapid[m.body(name).id]
        d.mocap_pos[mid] = [1.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        lp, rp = d.geom_xpos[m.geom(f"{name}_post_left").id], d.geom_xpos[m.geom(f"{name}_post_right").id]
        r = float(m.geom_size[m.geom(f"{name}_post_left").id][0])
        yaw = math.atan2(d.xmat[m.body(name).id][3], d.xmat[m.body(name).id][0])
        g = sr.Goal.from_posts(lp[:2], rp[:2], r, yaw)
        assert g.opening == pytest.approx(0.210, abs=1e-6)       # 3 ball widths, Dillon 2026-11-14
        assert g.post_radius == pytest.approx(0.010)
        assert g.to_frame(*lp[:2])[1] > 0   # the "left" post really is on +y
