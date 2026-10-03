from pathlib import Path
import tempfile
import unittest
import numpy as np

from piper_titration.checkpointing import initialize_checkpoint_dir


class CheckpointingTests(unittest.TestCase):
    def test_synchronous_composite_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            manager, resuming = initialize_checkpoint_dir(
                root, keep_period=None, overwrite=False, resume=False)
            self.assertFalse(resuming)
            try:
                params = {'params': {'weight': np.arange(12, dtype=np.float32)}}
                state = {'step': np.array(1), 'moment': np.ones(4, dtype=np.float32)}
                def assets(path):
                    (path/'marker.txt').write_text('asset')
                manager.save(1, {'assets': assets, 'params': params, 'train_state': state})
                self.assertTrue((root/'1/assets/marker.txt').is_file())
                self.assertTrue((root/'1/params').is_dir())
                restored = manager.restore(1, items={'params': params, 'train_state': state})
                np.testing.assert_array_equal(restored['params']['params']['weight'], params['params']['weight'])
                np.testing.assert_array_equal(restored['train_state']['moment'], state['moment'])
            finally:
                manager.close()
            with self.assertRaises(FileExistsError):
                initialize_checkpoint_dir(root, keep_period=None, overwrite=False, resume=False)
