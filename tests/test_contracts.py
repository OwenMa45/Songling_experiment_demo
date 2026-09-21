import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from piper_titration.episodes import Recorder, read, partition
from piper_titration.transforms import Inputs, Outputs


class ContractTests(unittest.TestCase):
    def observation(self):
        return {"state": np.arange(14, dtype=np.float32), "image": np.zeros((24, 24, 3), np.uint8), "wrist_image": np.ones((24, 24, 3), np.uint8), "prompt": "Dispense 3 drops"}

    def test_train_inference_image_conventions_match(self):
        obs = self.observation()
        live = Inputs()(obs)
        dataset = dict(obs, image=np.moveaxis(obs["image"], -1, 0)/255, wrist_image=np.moveaxis(obs["wrist_image"], -1, 0)/255)
        converted = Inputs()(dataset)
        for key in live["image"]:
            np.testing.assert_array_equal(live["image"][key], converted["image"][key])
        self.assertFalse(live["image_mask"]["left_wrist_0_rgb"])
        np.testing.assert_array_equal(live["state"], obs["state"])
        self.assertEqual(Outputs()({"actions": np.zeros((16, 32))})["actions"].shape, (16, 14))

    def test_record_preserves_before_state_and_applied_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = self.observation()
            recorder = Recorder(Path(tmp)/"episode", {"fps": 20})
            recorder.add(obs, obs["state"]+.1, 0)
            recorder.add(obs, obs["state"]+.2, .05)
            recorder.finish({"success": True})
            meta, rows = read(recorder.path)
            rows = list(rows)
            np.testing.assert_allclose(rows[0]["action"], obs["state"]+.1)
            np.testing.assert_array_equal(rows[0]["state"], obs["state"])
            self.assertEqual(meta["frames"], 2)

    def test_irregular_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = self.observation()
            r = Recorder(Path(tmp)/"episode", {"fps": 20})
            r.add(obs, obs["state"], 0)
            r.add(obs, obs["state"], .2)
            r.finish({})
            with self.assertRaises(ValueError):
                list(read(r.path)[1])

    def test_episode_split_has_no_overlap(self):
        paths = [Path(f"run_{i}/episode") for i in range(10)]
        a, b = partition(paths), partition(list(reversed(paths)))
        self.assertEqual(a, b)
        self.assertFalse(set(a["train"]) & set(a["test"]))
        self.assertEqual(len(a["test"]), 2)


if __name__ == "__main__":
    unittest.main()
