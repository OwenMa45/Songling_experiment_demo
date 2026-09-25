import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from piper_titration.capture_dataset import plan_dataset

spec=importlib.util.spec_from_file_location('prepared_auditor',Path(__file__).resolve().parents[1]/'scripts/audit_capture.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreparedTests(unittest.TestCase):
    def test_evidence_hash_and_hold_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config={'firmware_profile':'v189','can_interface':'can0','topology':'shared'}
            identity={**config,'result':'pass','verified':True}
            hold={'result':'pass','verified_configuration':config,
                  'conclusion':{'sample_and_hold_empirically_verified':True,'max_empirically_observed_hold_gap_ms':2400},
                  'dataset_policy':{'max_internal_hold_ms':1000,'max_terminal_hold_ms':1500}}
            for name, data in [('identity',identity),('hold',hold)]:
                (root/name).write_text(json.dumps(data))
            digest=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            meta={**config,'physical_feedback_source_verification':{'record':'identity','sha256':digest('identity')},
                  'action_source_verification':{'hold_behavior_record':'hold','hold_behavior_sha256':digest('hold'),'no_future_state_as_action':True},
                  'action_source_verified':True,'human_review':{'reviewed':True,'visual_ok':True}}
            row={'sample_index':0,'source_sample_index':12,'images':{},'action_provenance':{
                'approval':'verified_terminal_sample_and_hold','command_oldest_part_age_ms':1400,
                'command_complete_age_ms':1399,'target_part_skew_ms':1,
                'hold_verification_record_sha256':digest('hold'),'no_future_state_inference':True,
                'source_can_lines':{'155':1,'156':2,'157':3,'159':4}}}
            manifest=[]
            for camera in ('front','wrist'):
                (root/camera).write_bytes(b'test-image-bytes')
                row['images'][camera]={'path':camera}
                manifest.append({'sample_index':0,'camera':camera,'prepared_path':camera,'sha256':digest(camera)})
            (root/'images_manifest.jsonl').write_text('\n'.join(json.dumps(e) for e in manifest))
            self.assertEqual(m.prepared_checks(root,meta,[row]),[])
            row['action_provenance']['command_oldest_part_age_ms']=1501
            self.assertTrue(m.prepared_checks(root,meta,[row]))
            row['action_provenance']['command_oldest_part_age_ms']=1400
            (root/'identity').write_text('{}')
            self.assertIn('hash mismatch',m.prepared_checks(root,meta,[row])[0])

    def test_train_only_is_explicit_and_missing_episodes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); episode=root/'episode'; episode.mkdir()
            (episode/'metadata.json').write_text(json.dumps({'state_order':['q1','q2','q3','q4','q5','q6','gripper_width'],'cameras':{'front':'f','wrist':'w'}}))
            (episode/'samples.jsonl').write_text('{}\n')
            manifest=root/'selection.json'
            manifest.write_text(json.dumps({'episodes':[{'path':'episode','task_id':'beaker_move','group':'siteA'}]}))
            with patch('piper_titration.capture_dataset.audit_episode',return_value={'training_ready':True,'samples_sha256':'unique'}):
                self.assertEqual(set(plan_dataset(manifest,train_only=True)),{'train'})
                with self.assertRaisesRegex(ValueError,'three independent'):
                    plan_dataset(manifest)
            (episode/'samples.jsonl').unlink()
            with self.assertRaisesRegex(ValueError,'episode_files_missing'):
                plan_dataset(manifest,train_only=True)
