import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("recover_can_candidates", Path(__file__).resolve().parents[1]/"scripts/recover_can_candidates.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class CanRecoveryTests(unittest.TestCase):
    def frame(self, identifier, time=0, payload=None):
        if payload is None:
            payload = bytes.fromhex("0000000000000100") if identifier == 0x159 else bytes(8)
        return m.Frame(time, "can0", identifier, payload, time+1)

    def test_signed_units(self):
        values = m.decode(self.frame(0x155, payload=(-90000).to_bytes(4,"big",signed=True)+(180000).to_bytes(4,"big")))
        self.assertAlmostEqual(values[0], -math.pi/2)
        self.assertAlmostEqual(values[1], math.pi)
        self.assertAlmostEqual(m.decode(self.frame(0x159,payload=(45000).to_bytes(4,"big")+bytes.fromhex("00000100")))[0], .045)

    def test_reject_unsupported_gripper_and_payload(self):
        for identifier, payload in [(0x159,bytes(8)), (0x159,bytes.fromhex("0000000000000101")), (0x2A8,bytes.fromhex("0000000000000001")), (0x155,bytes(7))]:
            with self.subTest(identifier=identifier,payload=payload), self.assertRaises(ValueError):
                m.decode(self.frame(identifier,payload=payload))

    def test_no_cross_cycle_splicing(self):
        frames = [self.frame(i,t) for t,i in enumerate([0x155,0x156,0x155,0x157,0x159])]
        groups,rejected = m.target_groups(frames)
        self.assertEqual(groups, [])
        self.assertEqual(rejected["incomplete_group"], 1)

    def test_complete_cycle_and_span(self):
        frames = [self.frame(i,t*1000) for t,i in enumerate(m.TARGET_IDS)]
        groups,_ = m.target_groups(frames,max_span_ns=3000)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["action"]), 7)
        self.assertEqual(m.target_groups(frames,max_span_ns=2999)[0], [])

    def test_causal_alignment_and_oldest_frame_age(self):
        groups = [{"start_ns":100,"end_ns":110},{"start_ns":200,"end_ns":210}]
        ends = [110,210]
        self.assertEqual(m.match_group(groups,ends,109,100)[1], "no_complete_preceding_group")
        self.assertIs(m.match_group(groups,ends,200,100)[0], groups[0])
        self.assertEqual(m.match_group(groups,ends,201,100)[1], "target_group_stale")

    def test_log_rejects_rollback_wrong_bus_and_short_frame(self):
        for log in ["(2.0) can0 155#0000000000000000\n(1.0) can0 156#0000000000000000\n", "(1.0) can1 155#0000000000000000\n", "(1.0) can0 155#00\n"]:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"can.log"
                path.write_text(log)
                with self.assertRaises(ValueError):
                    m.read_frames(path)


if __name__ == "__main__":
    unittest.main()
