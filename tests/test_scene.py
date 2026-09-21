from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from piper_titration.scene import build


class SceneTests(unittest.TestCase):
    def test_external_meshes_and_actuator_limits(self):
        cfg = {"paths": {"piper_root": str(Path(__file__).resolve().parents[1]/"third_party/piper_ros")}, "simulation": {"timestep": .002, "image_size": 224, "holder_base": [0, -.3, 0], "squeezer_base": [0, .3, 0], "tube_center": [.35, 0, .14], "tube_radius": .014}}
        xml = ET.fromstring(build(cfg))
        for mesh in xml.findall("./asset/mesh"):
            self.assertTrue(Path(mesh.get("file")).is_file())
        joints = {j.get("name"): j for j in xml.findall(".//joint")}
        self.assertEqual(len(joints), 16)
        actuators = xml.findall("./actuator/position")
        self.assertEqual(len(actuators), 16)
        for actuator in actuators:
            self.assertEqual(actuator.get("ctrlrange"), joints[actuator.get("joint")].get("range"))
        self.assertEqual(len(xml.findall(".//camera")), 2)
        self.assertEqual(len(xml.findall(".//site[@name='tip']")), 1)


if __name__ == "__main__":
    unittest.main()
