"""Exercise CLI backend selection in fresh processes, before JAX initializes."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class NormBackendTests(unittest.TestCase):
    def invoke(self, command, body):
        root = Path(__file__).resolve().parents[1]
        code = '''
import sys
from unittest.mock import patch
from piper_titration.cli import main
def probe(*args, **kwargs):
''' + '\n'.join('    ' + line for line in body.splitlines()) + '''
with patch('piper_titration.config.load', return_value={'robot': {}}), \
     patch('piper_titration.training.execute', side_effect=probe):
    sys.argv = ['piper', ''' + repr(command) + ''']
    main()
'''
        env = dict(os.environ, PYTHONPATH=str(root/'src'), JAX_PLATFORMS='cuda')
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_norms_batch_one_works_with_gpu_backend_requested(self):
        self.invoke('norms', '''import jax
import numpy as np
assert jax.default_backend() == 'cpu'
devices = jax.devices()
assert len(devices) == 1, devices
sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, ('B',)), jax.sharding.PartitionSpec('B'))
batch = jax.make_array_from_process_local_data(sharding, np.zeros((1, 16, 7), np.float32))
assert batch.shape == (1, 16, 7)''')

    def test_training_keeps_requested_gpu_backend(self):
        self.invoke('train', '''import os
assert os.environ['JAX_PLATFORMS'] == 'cuda' ''')
