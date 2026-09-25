import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("audit_capture", Path(__file__).resolve().parents[1]/"scripts/audit_capture.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CaptureAuditTests(unittest.TestCase):
    def fixture(self, root, action=None):
        meta = {"samples": 2, "physical_feedback_source_verified": False,
                "ready_for_policy_training": False,"task_success":None,
                "state_units":["rad"]*6+["m"],"requested_fps":30,"cameras":{"front":"camera"}}
        rows = [{"host_monotonic_ns":i*33333333,"observation.state":[0.0]*7,
                 "action":action,"images":{"front":{"path":f"images/{i}.jpg", "sequence":i}}} for i in range(2)]
        (root/"metadata.json").write_text(json.dumps(meta))
        (root/"samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    def test_observation_only_is_not_command_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            report = module.audit(root)
            self.assertFalse(report["training_ready"])
            self.assertEqual(report["null_action_count"],2)
            self.assertIn("physical_feedback_source_not_verified",report["blockers"])
            self.assertIn("missing_or_invalid_command_actions",report["blockers"])
            self.assertEqual(len(report["missing_images"]),2)

    def test_filled_actions_do_not_override_unverified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root,[0.0]*7)
            report = module.audit(root)
            self.assertEqual(report["invalid_action_count"],0)
            self.assertFalse(report["training_ready"])
            self.assertIn("recording_explicitly_not_training_ready",report["blockers"])

    def test_audit_does_not_modify_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            module.audit(root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
