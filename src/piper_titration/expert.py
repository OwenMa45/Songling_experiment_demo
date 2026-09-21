import numpy as np


class ScriptedExpert:
    """One squeeze pulse followed by release and a settling interval."""
    def __init__(self, env):
        self.env = env
        self.phase = "position_holder"
        self.since = 0.0
        self.baseline = 0

    def action(self):
        e = self.env
        state = e.state()
        target = state.copy()
        target[:6] = e.hold_goal
        target[6] = 0.008
        target[7:13] = e.squeeze_start if self.phase == "position_holder" else e.squeeze_goal
        target[13] = 0.065
        now = e.data.time
        if self.phase == "position_holder" and np.max(abs(state[:6]-e.hold_goal)) < .025:
            self.phase, self.since = "approach", now
        elif self.phase == "approach" and np.linalg.norm(e.site("squeezer_tcp")-e.site("bulb_center")) < e.s["alignment_tolerance"]*.7:
            self.phase, self.since, self.baseline = "squeeze", now, e.counter.read()
        elif self.phase == "squeeze":
            target[13] = 0.009
            # Stop each pulse once one drop lands, or after a bounded attempt.
            if e.counter.read() > self.baseline or now-self.since > 2.5:
                self.phase, self.since = "release", now
                target[13] = 0.065
        elif self.phase == "release" and now-self.since > .8:
            self.phase, self.since, self.baseline = "squeeze", now, e.counter.read()
        return target
