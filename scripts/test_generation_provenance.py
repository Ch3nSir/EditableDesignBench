"""Offline generation identity checks; no model/GPU or network is used."""
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import generation_provenance as provenance


class GenerationContentTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.model=self.root/'model';self.model.mkdir()
        (self.model/'model_index.json').write_text('{"_class_name":"QwenImage21Pipeline"}')
        self.weight=self.model/'model.safetensors';self.weight.write_bytes(b'original weights')
        self.packages={}
        for name in ('diffusers','transformers'):
            folder=self.root/name;folder.mkdir();(folder/'__init__.py').write_text('__version__="unchanged"\n')
            (folder/'pipeline.py').write_text('def inference(): return 1\n')
            self.packages[name]=folder
        self.dataset=self.root/'dataset';(self.dataset/'cases/a').mkdir(parents=True)
        Image.new('RGB',(32,32),'white').save(self.dataset/'cases/a/source.png')
        (self.dataset/'tasks.jsonl').write_text(json.dumps({'id':'a','image':'cases/a/source.png',
                                      'instruction':"Delete 'A'."})+'\n')
        self.output=self.root/'output'

    def tearDown(self):self.temporary.cleanup()

    def collect(self,model=None):
        return provenance.collect_generation_content(model or self.model,package_roots=self.packages)


    def test_unchanged_content_is_stable_and_inventory_contains_weights_and_code(self):
        first=self.collect();second=self.collect()
        self.assertEqual(first,second)
        self.assertIn('model.safetensors',{r['path'] for r in first['model']['files']})
        self.assertEqual(first['model']['total_bytes'],sum(p.stat().st_size for p in self.model.iterdir()))
        for name in self.packages:
            self.assertEqual({r['path'] for r in first['python_packages'][name]['files']},
                             {'__init__.py','pipeline.py'})

    def test_changed_weight_content_same_size_and_mtime_changes_identity(self):
        first=self.collect();info=self.weight.stat()
        self.weight.write_bytes(b'modified weights')
        os.utime(self.weight,ns=(info.st_atime_ns,info.st_mtime_ns))
        self.assertEqual(self.weight.stat().st_size,info.st_size)
        self.assertNotEqual(first['sha256'],self.collect()['sha256'])

    def test_same_package_version_but_changed_source_changes_identity(self):
        first=self.collect();code=self.packages['diffusers']/'pipeline.py'
        code.write_text('def inference(): return 2\n')
        second=self.collect()
        self.assertEqual(first['python_packages']['diffusers']['version'],
                         second['python_packages']['diffusers']['version'])
        self.assertNotEqual(first['sha256'],second['sha256'])

    def test_runtime_model_config_and_tokenizer_files_are_hashed(self):
        tokenizer=self.model/'tokenizer';tokenizer.mkdir()
        config=tokenizer/'vocab.json';config.write_text('{"hello":1}')
        first=self.collect();config.write_text('{"hello":2}')
        self.assertNotEqual(first['model']['sha256'],self.collect()['model']['sha256'])

    def test_missing_referenced_checkpoint_shard_fails(self):
        (self.model/'model.safetensors.index.json').write_text(json.dumps({
            'weight_map':{'layer.weight':'missing-00001.safetensors'}}))
        with self.assertRaisesRegex(ValueError,'Missing/invalid model shard'):self.collect()

    def test_symlinked_weight_content_is_hashed_and_cycles_rejected(self):
        external=self.root/'external';external.mkdir();(external/'part.safetensors').write_bytes(b'A')
        (self.model/'component').symlink_to(external,target_is_directory=True)
        first=self.collect();(external/'part.safetensors').write_bytes(b'B')
        self.assertNotEqual(first['sha256'],self.collect()['sha256'])
        (external/'cycle').symlink_to(self.model,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlink directory cycle'):self.collect()







if __name__=='__main__':unittest.main()
