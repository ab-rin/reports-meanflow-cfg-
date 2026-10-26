"""Check that batch migration cannot silently alter full-state training settings."""
import copy
import unittest
from unittest.mock import patch

from train import parse_args,validate_resume_config


class ResumeOptionsTests(unittest.TestCase):
    def setUp(self):
        # Legacy checkpoint metadata predates the migration flag.
        self.saved=dict(microbatch=16,effective_batch=512,world_size=1,
                        gradient_accumulation_per_gpu=32,updates=100000,fid_every=1000,
                        fid_batch=32,fid_seed=1234,fid_reference='official_validation',
                        lr=3e-4,ema_decay=.999,guidance_scale=2.,seed=42,
                        optimizer_initialization='fresh Adam; randomly initialized model')
        self.current={**self.saved,'allow_microbatch_change':False,
                      'optimizer_initialization':'Adam state restored from full-state checkpoint'}

    def test_ordinary_resume_accepts_legacy_checkpoint(self):
        changes=validate_resume_config(self.saved,self.current)
        self.assertEqual(changes['optimizer_initialization']['new'],
                         'Adam state restored from full-state checkpoint')
        self.assertNotIn('microbatch',changes)

    def test_resume_after_migration_does_not_require_flag_again(self):
        previous={**self.current,'microbatch':32,'gradient_accumulation_per_gpu':16,
                  'allow_microbatch_change':True}
        current={**previous,'allow_microbatch_change':False}
        self.assertEqual(validate_resume_config(previous,current),
                         {'allow_microbatch_change':{'old':True,'new':False}})

    def test_migration_requires_explicit_opt_in(self):
        for microbatch in (32,64):
            with self.subTest(microbatch=microbatch):
                current={**self.current,'microbatch':microbatch,
                         'gradient_accumulation_per_gpu':512//microbatch}
                with self.assertRaisesRegex(ValueError,'Resume mismatch: microbatch'):
                    validate_resume_config(self.saved,current)

    def test_opt_in_allows_only_derived_accumulation_change(self):
        for microbatch in (32,64):
            with self.subTest(microbatch=microbatch):
                current={**self.current,'microbatch':microbatch,
                         'gradient_accumulation_per_gpu':512//microbatch}
                original=copy.deepcopy(self.saved)
                changes=validate_resume_config(self.saved,current,allow_microbatch_change=True)
                self.assertEqual(changes['microbatch'],{'old':16,'new':microbatch})
                self.assertEqual(changes['gradient_accumulation_per_gpu'],
                                 {'old':32,'new':512//microbatch})
                self.assertEqual(self.saved,original)

    def test_effective_batch_change_is_always_rejected(self):
        current={**self.current,'effective_batch':1024,'microbatch':32,
                 'gradient_accumulation_per_gpu':32}
        with self.assertRaisesRegex(ValueError,'Resume mismatch: effective_batch'):
            validate_resume_config(self.saved,current,allow_microbatch_change=True)

    def test_world_size_change_is_always_rejected(self):
        current={**self.current,'world_size':2,'microbatch':32,
                 'gradient_accumulation_per_gpu':8}
        with self.assertRaisesRegex(ValueError,'Resume mismatch: world_size'):
            validate_resume_config(self.saved,current,allow_microbatch_change=True)

    def test_inconsistent_or_indivisible_accumulation_is_rejected(self):
        for microbatch,accumulation in ((32,32),(48,10),(0,0)):
            with self.subTest(microbatch=microbatch,accumulation=accumulation):
                current={**self.current,'microbatch':microbatch,
                         'gradient_accumulation_per_gpu':accumulation}
                with self.assertRaisesRegex(ValueError,'Invalid requested'):
                    validate_resume_config(self.saved,current,allow_microbatch_change=True)
        saved={**self.saved,'gradient_accumulation_per_gpu':16}
        with self.assertRaisesRegex(ValueError,'Invalid checkpoint'):
            validate_resume_config(saved,self.current,allow_microbatch_change=True)

    def test_training_and_fid_identity_remain_strict(self):
        for key,value in [('lr',6e-4),('ema_decay',.99),('guidance_scale',3.),
                          ('seed',43),('fid_seed',4321),('fid_reference','train_split')]:
            with self.subTest(key=key):
                current={**self.current,key:value,'microbatch':32,
                         'gradient_accumulation_per_gpu':16}
                with self.assertRaisesRegex(ValueError,f'Resume mismatch: {key}'):
                    validate_resume_config(self.saved,current,allow_microbatch_change=True)

    def test_150k_and_full_fid_every_250_are_runtime_overrides(self):
        current={**self.current,'updates':150000,'fid_every':250}
        changes=validate_resume_config(self.saved,current)
        self.assertEqual(changes['updates'],{'old':100000,'new':150000})
        self.assertEqual(changes['fid_every'],{'old':1000,'new':250})
        for key in ('fid_batch','fid_seed','fid_reference'):
            self.assertNotIn(key,changes)

    def test_cli_flag_defaults_to_false_and_can_be_enabled(self):
        argv=['train.py','--output','/tmp/resume-test','--updates','150000',
              '--resume','/tmp/source.pt','--fid-every','250']
        with patch('sys.argv',argv):
            self.assertFalse(parse_args().allow_microbatch_change)
        with patch('sys.argv',argv+['--allow-microbatch-change']):
            self.assertTrue(parse_args().allow_microbatch_change)


if __name__=='__main__':
    unittest.main()
