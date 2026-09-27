import json
from pathlib import Path
import tempfile
import unittest
from piper_titration.capture_dataset import validate_receipt
from piper_titration.single_arm import SCHEMA


class ReceiptDiagnosticsTests(unittest.TestCase):
    def test_mismatch_reports_both_ids_and_receipt_without_lerobot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            path.write_text(json.dumps({'schema':SCHEMA,'control_hz':30,
                'splits':{'train':{'repo_id':'local/exported_train'}}}))
            cfg={'paths':{'dataset_receipt':str(path)},'robot':{'control_hz':30},
                 'training':{'repo_id':'local/configured_train'}}
            before=path.read_bytes()
            with self.assertRaises(ValueError) as context:
                validate_receipt(cfg)
            message=str(context.exception)
            for value in ('local/exported_train','local/configured_train',str(path.resolve()),'PIPER_CONFIG'):
                self.assertIn(value,message)
            self.assertEqual(before,path.read_bytes())
