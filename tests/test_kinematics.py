"""Independent forward kinematics from upstream XML, without MuJoCo."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from piper_titration.kinematics import solve, tool_rotation


def quat_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class ReachabilityTests(unittest.TestCase):
    def test_default_tool_poses_reachable(self):
        root = Path(__file__).resolve().parents[1]
        model = ET.parse(root/"third_party/piper_ros/src/piper_description/mujoco_model/piper_description.xml")
        bodies = [model.find(f".//body[@name='link{i}']") for i in range(1, 7)]
        limits = np.array([np.fromstring(b.find("joint").get("range"), sep=" ") for b in bodies])
        for arm, base, rot, target in [
            ("holder", [0, -.30, 0], tool_rotation(.7), [.35, 0, .295]),
            ("squeezer", [0, .30, 0], tool_rotation(-.7), [.35, 0, .35]),
        ]:
            def forward(q):
                p, r = np.array(base, dtype=float), np.eye(3)
                for b, a in zip(bodies, q):
                    p = p+r@np.fromstring(b.get("pos", "0 0 0"), sep=" ")
                    r = r@quat_matrix(np.fromstring(b.get("quat", "1 0 0 0"), sep=" "))
                    r = r@np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                return p+r@np.array([0, 0, .18]), r
            q = solve(forward, np.array(target), np.array(rot), limits)
            self.assertLess(np.linalg.norm(forward(q)[0]-target), .004, arm)
            offset = [-.035, -.025, .03] if arm == "holder" else [0, .065, .035]
            start = solve(forward, np.array(target)+offset, np.array(rot), limits, q)
            self.assertLess(np.linalg.norm(forward(start)[0]-np.array(target)-offset), .004, arm)


if __name__ == "__main__":
    unittest.main()
