import importlib.util
from pathlib import Path
import sys
import unittest

scripts=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('train_remote_3090',scripts/'train_remote_3090.py')
remote=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=remote
spec.loader.exec_module(remote)
spec=importlib.util.spec_from_file_location('two_skills',scripts/'train_two_skills.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TwoSkillTests(unittest.TestCase):
    def test_task_receipt_rejects_mixed_data(self):
        receipt={'splits':{'train':{'episodes':[{'task_id':'beaker_move'}]}}}
        m.validate_task_receipt(receipt,{'beaker_move'})
        with self.assertRaises(ValueError):
            m.validate_task_receipt(receipt,{'tube_pour'})
        receipt['splits']['train']['episodes'].append({'task_id':'tube_pour'})
        with self.assertRaises(ValueError):
            m.validate_task_receipt(receipt,{'beaker_move'})

    def test_output_dataset_and_receipt_isolation(self):
        configs=[{'repo_id':'b','output':'b','receipt':'b'}, {'repo_id':'t','output':'t','receipt':'t'}]
        m.validate_isolation(configs)
        for field in configs[0]:
            bad=[dict(c) for c in configs]; bad[1][field]=bad[0][field]
            with self.assertRaises(ValueError):
                m.validate_isolation(bad)

    def test_remote_worker_not_old_beaker_train(self):
        args=remote.parser().parse_args(['doctor'])
        _,script=remote.build(args,worker_command=['scripts/train_two_skills.py','--worker','--stage','all'])
        self.assertIn('scripts/train_two_skills.py --worker --stage all',script)
        self.assertNotIn('train --steps',script)
