"""Additional profiles cannot redirect inputs or corrupt existing registrations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_registry as registry


class CustomModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'models.json'
        self.patch=patch.dict(registry.MODEL_CONFIGS)
        self.patch.start();self.addCleanup(self.patch.stop)
        self.profile=registry.get_model_config('qwen-edit-2511')
        self.profile.pop('local_dir')

    def load(self,models):
        self.path.write_text(json.dumps({'models':models}))
        registry.load_model_config(self.path)

    def test_example_is_loadable_and_pipeline_signature_matches_installed_source(self):
        registry.load_model_config(Path(__file__).resolve().parents[1]/'config/models.example.json')
        config=registry.get_model_config('my-qwen-edit')
        self.assertEqual(config['local_dir'],str(registry.MODEL_ROOT/'my-qwen-edit'))
        self.assertIn('image',registry.validate_pipeline_signature(config))
        registry.load_model_config(Path(__file__).resolve().parents[1]/'config/models.example.json')

    def test_invalid_file_does_not_partially_register_profiles(self):
        bad=copy.deepcopy(self.profile);bad['call_kwargs']['image']='reference.png'
        with self.assertRaisesRegex(ValueError,'cannot override'):
            self.load({'valid-model':self.profile,'bad-model':bad})
        self.assertNotIn('valid-model',registry.MODEL_CONFIGS)

    def test_existing_builtin_cannot_be_replaced_or_checkpoint_directory_shared(self):
        with self.assertRaisesRegex(ValueError,'replace'):
            self.load({'qwen-edit-2511':self.profile})
        self.profile['local_dir']=Path(registry.get_model_config('qwen-edit-2511')['local_dir']).name
        with self.assertRaisesRegex(ValueError,'distinct checkpoint'):
            self.load({'other-model':self.profile})

    def test_traversal_and_nonfinite_or_invalid_settings_are_rejected(self):
        for key,value in [('local_dir','../checkpoint'),('pipeline_source','../source.py'),
                          ('num_inference_steps',True),('required_components',['../target']),
                          ('padding_multiple',64),('call_kwargs',{'guidance_scale':float('nan')})]:
            profile=copy.deepcopy(self.profile);profile[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.load({'extra':profile})
        with self.assertRaisesRegex(ValueError,'Invalid custom model ID'):
            self.load({'../outside':self.profile})


if __name__=='__main__':unittest.main()
