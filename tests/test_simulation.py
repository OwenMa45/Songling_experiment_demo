import unittest
import numpy as np
try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class SimulationTests(unittest.TestCase):
    def make_env(self):
        from piper_titration.config import load
        from piper_titration.env import TitrationEnv
        env = TitrationEnv(load())
        self.addCleanup(env.close)
        return env

    def test_invalid_action_stops(self):
        env = self.make_env()
        env.step(np.full(14, np.nan))
        self.assertEqual(env.reason, "invalid_action")

    def test_scripted_target_and_residual_drops(self):
        from piper_titration.expert import ScriptedExpert
        env = self.make_env()
        expert = ScriptedExpert(env)
        while not env.done:
            env.step(expert.action())
        self.assertTrue(env.report()["success"], env.report())


if __name__ == "__main__":
    unittest.main()
