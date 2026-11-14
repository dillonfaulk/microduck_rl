"""Soccer referee: pure scoring logic for GoalKick missions now and a 2v2 match later.

No simulator imports here. The sim side (kick_mission.py) reads the goal pose and post
contacts from MuJoCo and feeds plain numbers in; tests drive it with synthetic tracks.

Conventions (same as the goal model, src/mjlab_microduck/robot/microduck/goal.xml):
  goal frame origin = centre of the goal line, +x = INTO the goal, +y = left when facing in.
  lateral > 0 = the ball passed LEFT of the goal centre as the kicker sees it (matches the
  missions' "+ = left" kick-direction sign).

Outcomes, one per kick (the GoalKick brief's lines, kept separate, never merged):
  "goal"     ball centre crossed the goal line between the posts' inner faces, no post contact
  "post"     the ball touched a post (went_in says whether it then crossed between the posts)
  "miss"     crossed the goal-line extension outside the posts, no post contact
  "no_cross" never reached the goal line before the timeout (no post contact)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Goal:
    cx: float            # goal-line centre, world
    cy: float
    yaw: float           # direction INTO the goal, world rad
    half_opening: float  # centre to a post's inner face, m (0.105 for the 210 mm goal)
    post_radius: float

    @classmethod
    def from_posts(cls, left_xy, right_xy, post_radius, into_yaw):
        """Build from the two post centres (world xy) as the compiled model reports them."""
        cx, cy = (left_xy[0] + right_xy[0]) / 2, (left_xy[1] + right_xy[1]) / 2
        half_span = math.hypot(left_xy[0] - right_xy[0], left_xy[1] - right_xy[1]) / 2
        return cls(cx, cy, into_yaw, half_span - post_radius, post_radius)

    def to_frame(self, x, y):
        """World xy -> (along, lateral) in the goal frame. along > 0 = past the goal line."""
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    @property
    def opening(self):
        return 2 * self.half_opening


@dataclass(frozen=True)
class Field:
    """Axis-aligned pitch for the out-of-bounds geofence (2v2 match; unused by GoalKick v0)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def out_of_bounds(self, x, y, ball_radius=0.0):
        """Soccer rule: out only when the WHOLE ball is over the line."""
        r = ball_radius
        return x < self.x_min - r or x > self.x_max + r or y < self.y_min - r or y > self.y_max + r


class GoalReferee:
    """Scores one kick at one goal. Feed it every control tick from the kick swap-in on."""

    def __init__(self, goal: Goal, timeout_s: float = 10.0):
        self.goal = goal
        self.timeout_s = timeout_s
        self.t0 = None
        self.prev = None          # (t, along, lateral) of the last tick
        self.post_hit = False
        self.t_post = None
        self.result = None        # set once, then frozen

    @property
    def done(self):
        return self.result is not None

    def update(self, t, ball_x, ball_y, ball_post_contact=False):
        """One tick. Returns the result dict once the kick is decided, else None."""
        if self.result is not None:
            return self.result
        if self.t0 is None:
            self.t0 = t
        if ball_post_contact and not self.post_hit:
            self.post_hit, self.t_post = True, round(t - self.t0, 2)
        along, lat = self.goal.to_frame(ball_x, ball_y)
        if self.prev is not None:
            tp, ap, lp = self.prev
            if ap < 0.0 <= along:   # crossed the goal line (or its extension) heading in
                f = -ap / (along - ap) if along != ap else 1.0
                lat_x = lp + f * (lat - lp)
                t_x = tp + f * (t - tp)
                inside = abs(lat_x) < self.goal.half_opening
                self._decide(("post" if self.post_hit else "goal") if inside else ("post" if self.post_hit else "miss"),
                             lateral=lat_x, t_line=t_x - self.t0, went_in=inside)
                return self.result
        self.prev = (t, along, lat)
        if t - self.t0 >= self.timeout_s:
            self._decide("post" if self.post_hit else "no_cross", lateral=None, t_line=None, went_in=False)
            return self.result
        return None

    def _decide(self, outcome, lateral, t_line, went_in):
        miss_by = None
        if lateral is not None and not went_in:
            miss_by = abs(lateral) - self.goal.half_opening   # clear distance outside the post's inner face
        self.result = {
            "goal_outcome": outcome,
            "goal_lateral_mm": None if lateral is None else round(lateral * 1000, 1),
            "goal_miss_by_mm": None if miss_by is None else round(miss_by * 1000, 1),
            "goal_t_line_s": None if t_line is None else round(t_line, 2),
            "goal_post_hit": self.post_hit,
            "goal_post_t_s": self.t_post,
            "goal_went_in": went_in,
        }
