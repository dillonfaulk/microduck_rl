"""Rung 6 Stage 4, Workstream C: ground truth wired to SCORING only, and proven.

`TruthGuard(sim)` is what the supervisor (kick_mission.Mission) is handed in an estimated-only run:
everything on the sim passes through (policy, data for proprioception / FK, contacts for the referee),
except the two ground-truth accessors, `pose()` and `ball()`, which raise `TruthAccess` unless the
caller is inside `allow_truth()` -- the context Mission._truth() opens around its scoring and log
reads. The sensor models (pose providers, camera ball/goal, mapd shim) are built on the RAW sim: they
are the declared boundary where the simulator stands in for the robot's sensors.
"""
from contextlib import contextmanager


class TruthAccess(RuntimeError):
    """A supervisor input was sourced from sim state."""


class TruthGuard:
    def __init__(self, sim):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_depth", 0)
        object.__setattr__(self, "scoring_reads", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_sim"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_sim"), name, value)

    @contextmanager
    def allow_truth(self):
        object.__setattr__(self, "_depth", self._depth + 1)
        try:
            yield
        finally:
            object.__setattr__(self, "_depth", self._depth - 1)

    def _check(self, what):
        if self._depth <= 0:
            raise TruthAccess(f"supervisor read sim.{what}() outside scoring: a control input sourced from sim state")
        object.__setattr__(self, "scoring_reads", self.scoring_reads + 1)

    def pose(self):
        self._check("pose")
        return self._sim.pose()

    def ball(self):
        self._check("ball")
        return self._sim.ball()
