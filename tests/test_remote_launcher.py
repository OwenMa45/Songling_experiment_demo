import importlib.util
from pathlib import Path
import shlex
import unittest

spec=importlib.util.spec_from_file_location('remote3090',Path(__file__).resolve().parents[1]/'scripts/train_remote_3090.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RemoteLauncherTests(unittest.TestCase):
    def test_smoke_and_existing_receipt(self):
        args=m.parser().parse_args(['smoke','--identity','/keys/my key','--experiment','check'])
        cmd,script=m.build(args)
        self.assertIn('lzk@10.71.106.251',cmd)
        self.assertIn('/keys/my key',cmd)
        self.assertIn('StrictHostKeyChecking=yes',cmd)
        self.assertIn('outputs/chemical/dataset_receipt.json',script)
        self.assertIn('--steps 1 --batch-size 4 --fsdp-devices 4',script)
        self.assertIn('set -euo pipefail',script)

    def test_invalid_layout_rejected(self):
        for options in (['--gpu-list','4'],['--gpu-list','0,0'],['--batch-size','1'],['--fsdp-devices','3']):
            with self.subTest(options=options),self.assertRaises(ValueError):
                m.build(m.parser().parse_args(['smoke']+options))

    def test_shell_paths_quoted_and_python_override(self):
        path='/tmp/a $(touch BAD)'
        cmd,script=m.build(m.parser().parse_args(['doctor','--project',path,'--python','/env/bin/python']))
        self.assertIn('cd -- '+shlex.quote(path),script)
        self.assertNotIn('conda run',script)
        self.assertIn('/env/bin/python -m piper_titration',script)
        self.assertNotIn('train --steps',script)
