"""Staged checkpoint-presence tests; auto-enable when the reviewed patch lands.

The production registry is frozen while current GPU jobs run. Keeping these
tests skipped until its helper exists prevents claiming that this stricter
preflight is already implemented.
"""
import json
from pathlib import Path
import tempfile
import unittest

import model_registry as registry


@unittest.skipUnless(hasattr(registry,'_validate_component_checkpoint'),
                     'Checkpoint presence patch is staged until active GPU jobs finish')
class ModelCheckpointPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config={'pipeline':'FixturePipeline','required_components':['transformer','vae','text_encoder']}
        index={'_class_name':'FixturePipeline'}
        for component in self.config['required_components']:
            folder=self.root/component;folder.mkdir();(folder/'config.json').write_text('{}')
            (folder/'model.safetensors').write_bytes(b'fixture checkpoint bytes')
            index[component]=['diffusers','FixtureComponent']
        (self.root/'model_index.json').write_text(json.dumps(index))

    def tearDown(self):self.temp.cleanup()

    def test_complete_components_pass(self):
        self.assertEqual(registry.validate_model_files(self.root,self.config)['_class_name'],'FixturePipeline')

    def test_metadata_without_text_encoder_weights_is_not_ready(self):
        (self.root/'text_encoder/model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'checkpoint weights missing'):
            registry.validate_model_files(self.root,self.config)

    def test_zero_byte_checkpoint_is_not_ready(self):
        (self.root/'vae/model.safetensors').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'checkpoint weights missing'):
            registry.validate_model_files(self.root,self.config)

    def test_missing_second_shard_is_rejected_before_model_load(self):
        folder=self.root/'transformer'
        (folder/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model.safetensors','b':'missing.safetensors'}}))
        with self.assertRaisesRegex(ValueError,'checkpoint shard missing'):
            registry.validate_model_files(self.root,self.config)

    def test_unsafe_or_invalid_shard_reference_is_rejected(self):
        path=self.root/'transformer/model.safetensors.index.json'
        for value in ('../outside.safetensors','/outside.safetensors',123):
            path.write_text(json.dumps({'weight_map':{'a':value}}))
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'shard path'):
                registry.validate_model_files(self.root,self.config)

    def test_complete_indexed_checkpoint_passes(self):
        folder=self.root/'transformer';(folder/'part2.safetensors').write_bytes(b'other fixture shard')
        (folder/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model.safetensors','b':'part2.safetensors'}}))
        registry.validate_model_files(self.root,self.config)

    def test_checkpoint_index_requires_nonempty_weight_map(self):
        path=self.root/'transformer/model.safetensors.index.json'
        for data in ({},{'weight_map':[]},{'weight_map':{}}):
            path.write_text(json.dumps(data))
            with self.subTest(data=data),self.assertRaisesRegex(ValueError,'Invalid checkpoint weight_map'):
                registry.validate_model_files(self.root,self.config)


if __name__=='__main__':unittest.main()
