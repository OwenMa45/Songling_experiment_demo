import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from piper_titration.norm_audit import (inspect_stats, dataset_fingerprint, binding,
    norm_path, stamp_path, sha256, check_norms, check_checkpoint, roundtrip_arrays)
from piper_titration.single_arm import SCHEMA, ORDER, coordinate_contract, data_transforms
from piper_titration.capture_dataset import validate_features
from piper_titration.transforms import rgb


class NormAuditTests(unittest.TestCase):
    def setup_case(self, root):
        cfg={'robot':{'schema':SCHEMA,'action_dim':7,'control_hz':30},
             'training':{'action_horizon':16},'paths':{'dataset_receipt':str(root/'receipt.json')}}
        config=SimpleNamespace(assets_dirs=root/'assets',data=SimpleNamespace(repo_id='local/demo'))
        dataset=root/'dataset'; (dataset/'data').mkdir(parents=True); (dataset/'meta').mkdir()
        (dataset/'data/episode.parquet').write_bytes(b'numeric supervision fixture')
        (dataset/'meta/info.json').write_text('{}')
        receipt={'splits':{'train':{'root':str(dataset)}}}
        (root/'receipt.json').write_text(json.dumps(receipt))
        stats={k:{'mean':[0.]*7,'std':[1.]*7,'q01':[-1.]*7,'q99':[1.]*7} for k in ('state','actions')}
        norm_path(config).parent.mkdir(parents=True)
        norm_path(config).write_text(json.dumps({'norm_stats':stats}))
        report={'binding':binding(cfg,config,receipt),'norm_sha256':sha256(norm_path(config)),
                'stats':inspect_stats(norm_path(config))}
        stamp_path(config).write_text(json.dumps(report))
        return cfg,config,receipt,report

    def test_modified_data_horizon_and_norms_cannot_silently_reuse_stats(self):
        for mode in ('data','horizon','norm','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); cfg,config,receipt,_=self.setup_case(root)
                check_norms(cfg,config,receipt)
                if mode=='data': (root/'dataset/data/episode.parquet').write_bytes(b'changed supervision')
                elif mode=='horizon': cfg['training']['action_horizon']=32
                elif mode=='norm': norm_path(config).write_text('{}')
                else: stamp_path(config).unlink()
                with self.assertRaises(ValueError): check_norms(cfg,config,receipt)

    def test_stats_dimensions_quantiles_and_degenerate_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,config,receipt,report=self.setup_case(Path(tmp))
            stats=json.loads(norm_path(config).read_text())
            stats['norm_stats']['actions']['q99'][0]=-1
            norm_path(config).write_text(json.dumps(stats))
            self.assertEqual(inspect_stats(norm_path(config))['actions']['near_constant_coordinates'],['q1'])
            stats['norm_stats']['actions']['q99'][0]=-2
            norm_path(config).write_text(json.dumps(stats))
            with self.assertRaises(ValueError): inspect_stats(norm_path(config))
            stats['norm_stats']['actions']['q99']=[1]*14
            norm_path(config).write_text(json.dumps(stats))
            with self.assertRaises(ValueError): inspect_stats(norm_path(config))

    def test_checkpoint_uses_saved_numeric_norms_even_if_json_format_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg,config,receipt,report=self.setup_case(root)
            checkpoint=root/'step'
            asset=checkpoint/'assets/local/demo/norm_stats.json'; asset.parent.mkdir(parents=True)
            stats=json.loads(norm_path(config).read_text()); asset.write_text(json.dumps(stats,indent=4))
            check_checkpoint(cfg,config,checkpoint,report)
            stats['norm_stats']['actions']['mean'][0]=1
            asset.write_text(json.dumps(stats))
            with self.assertRaises(ValueError): check_checkpoint(cfg,config,checkpoint,report)

    def test_image_units_and_channel_layout(self):
        rgb(np.ones((3,12,16),dtype=np.float32))
        rgb(np.full((12,16,3),255,dtype=np.uint8))
        for image in (np.full((12,16,3),255.),np.full((12,16,3),np.nan),np.ones((12,16,3),dtype=np.int32)):
            with self.assertRaises(ValueError): rgb(image)

    def test_lerobot_order_and_frequency(self):
        info={'fps':30,'features':{k:{'shape':[7],'dtype':'float32','names':ORDER} for k in ('state','actions')}}
        validate_features(info,30)
        with self.assertRaises(ValueError): validate_features(info,20)
        info['features']['actions']['names']=list(reversed(ORDER))
        with self.assertRaises(ValueError): validate_features(info,30)

    def test_official_roundtrip_when_openpi_environment_available(self):
        try:
            from openpi.shared.normalize import NormStats
        except ImportError:
            self.skipTest('Pinned openpi environment not installed locally; norm-check runs this path on NAS')
        state=np.array([.1,-.2,.3,.4,-.5,.6,.02],np.float32)
        actions=np.stack([state+.05,state-.1]); actions[:,6]=[-.0026,.0519]
        original=actions.copy()
        stats={k:NormStats(mean=np.zeros(7),std=np.ones(7),q01=-np.ones(7),q99=np.ones(7)) for k in ('state','actions')}
        result,padded=roundtrip_arrays(state,actions,stats,data_transforms())
        np.testing.assert_allclose(result,actions,rtol=0,atol=2e-6)
        np.testing.assert_array_equal(actions,original)
        self.assertEqual(padded['actions'].shape,(2,32))


if __name__=='__main__': unittest.main()
