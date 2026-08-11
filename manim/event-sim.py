"""ManimGL entry point for the event-based particle transport demo."""

from manimlib import ThreeDScene

try:
    from demonstrations.event_sim import event_sim_scene
except ModuleNotFoundError:
    # Supports running `manimgl event-sim.py` from this directory.
    from event_sim import event_sim_scene


class EventSimDemo(ThreeDScene):
    always_depth_test = False

    def construct(self):
        event_sim_scene(self)
