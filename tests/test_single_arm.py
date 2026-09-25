import copy
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np
from piper_titration.single_arm import Inputs, Outputs, ORDER
from piper_titration.capture_dataset import plan_dataset, resample_indices
from piper_titration.config import load


class SingleArmTests(unittest.TestCase):
    def test_default_config_is_single_arm(self):
        self.assertEqual(load()["robot"]["action_dim"],7)
        self.assertEqual(load()["robot"]["control_hz"],30)

    def test_single_arm_mapping_does_not_flip_or_synthesize_joints(self):
        data = {"state":np.arange(7,dtype=float),"actions":np.ones((16,7)),
                "image":np.zeros((224,224,3),np.uint8),"wrist_image":np.ones((224,224,3),np.uint8),"prompt":"Pick up the beaker."}
        original = copy.deepcopy(data)
        result = Inputs()(data)
        np.testing.assert_array_equal(result["state"],data["state"])
        result["actions"][:] = 0
        np.testing.assert_array_equal(data["actions"],original["actions"])
        self.assertFalse(result["image_mask"]["left_wrist_0_rgb"])
        self.assertEqual(Outputs()({"actions":np.zeros((16,32))})["actions"].shape,(16,7))
        data["state"] = np.zeros(14)
        with self.assertRaises(ValueError):
            Inputs()(data)

    def test_resampling_is_causal_and_rejects_long_gaps(self):
        rows = [{"host_monotonic_ns":t} for t in (0, 34000000, 66000000, 101000000)]
        indices, ages = resample_indices(rows,30)
        self.assertTrue(np.all(ages>=0))
        self.assertEqual(indices.tolist(),[0,0,2,2])
        with self.assertRaises(ValueError):
            resample_indices([{"host_monotonic_ns":0},{"host_monotonic_ns":1000000000}],30)

    def test_null_actions_reject_plan_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"selection.json").write_text(json.dumps({"episodes":[{"path":"episode","task_id":"beaker_move","group":"session1"}]}))
            with patch("piper_titration.capture_dataset.audit_episode",return_value={"training_ready":False,"blockers":["missing_or_invalid_command_actions"]}):
                with self.assertRaisesRegex(ValueError,"no dataset written"):
                    plan_dataset(root/"selection.json")
            self.assertEqual(list(root.iterdir()),[root/"selection.json"])

    def test_whole_collection_groups_do_not_cross_splits(self):
        self.check_split(shared=True)

    def test_small_independent_task_groups_can_form_splits(self):
        self.check_split(shared=False)

    def check_split(self, shared):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for task in ("beaker_move","test_tube_pour_return"):
                for group in range(10 if shared else 3):
                    episode = root/f"{task}_{group}"
                    episode.mkdir()
                    (episode/"samples.jsonl").write_text("{}\n")
                    (episode/"metadata.json").write_text(json.dumps({"state_order":ORDER,"cameras":{"front":"f","wrist":"w"}}))
                    group_id = f"session_{group}" if shared else f"{task}_session_{group}"
                    entries.append({"path":episode.name,"task_id":task,"group":group_id})
            path = root/"selection.json"
            path.write_text(json.dumps({"episodes":entries}))
            def checked(p):
                return {"training_ready":True,"samples_sha256":str(p)}
            with patch("piper_titration.capture_dataset.audit_episode",side_effect=checked):
                splits = plan_dataset(path)
            groups = {s:{e["group"] for e in v} for s,v in splits.items()}
            self.assertFalse(groups["train"]&groups["test"])
            self.assertFalse(groups["validation"]&groups["test"])
            self.assertFalse(groups["train"]&groups["validation"])
            for entries in splits.values():
                self.assertEqual({e["task_id"] for e in entries},{"beaker_move","test_tube_pour_return"})


if __name__ == "__main__":
    unittest.main()
