"""Portable source lookup is byte-identical and preserves existing local IDs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mllm_evaluate as judge
import official_pixel_metrics as pixels


class OfficialSourcePortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.original=self.root/'original';self.original.mkdir()
        for name in ('mse_psnr.py','lipis_ssim.py'):
            shutil.copyfile(pixels.BUNDLED_EVALUATION/name,self.original/name)
        self.prompts=json.loads(judge.PROMPT_FILE.read_text())['modes']

    def tearDown(self):self.temp.cleanup()

    def test_vendor_files_match_recorded_upstream_hashes_without_execution(self):
        root=pixels.BUNDLED_EVALUATION.parent
        manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
        self.assertEqual(len(manifest['files']),4)
        self.assertFalse(any(manifest['secret_scan'].values()))
        for name,record in manifest['files'].items():
            path=root/name
            self.assertEqual(path.stat().st_size,record['bytes'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])

    def test_pixel_prefers_original_when_available(self):
        with patch.object(pixels,'LEGACY_EVALUATION',self.original),patch.dict(os.environ,{},clear=True):
            self.assertEqual(pixels.resolve_official_evaluation(),self.original.resolve())

    def test_pixel_falls_back_to_identical_bundled_source_when_original_missing(self):
        with patch.object(pixels,'LEGACY_EVALUATION',self.root/'missing'),patch.dict(os.environ,{},clear=True):
            folder=pixels.resolve_official_evaluation()
            self.assertEqual(folder,pixels.BUNDLED_EVALUATION.resolve())
            protocol=pixels.protocol_info()
        for name,record in protocol['sources'].items():
            self.assertEqual(record['sha256'],hashlib.sha256((self.original/name).read_bytes()).hexdigest())

    def test_explicit_pixel_directory_and_environment_override_are_checked(self):
        with patch.dict(os.environ,{'TEXTEDITBENCH_EVALUATION_DIR':str(self.original)}):
            self.assertEqual(pixels.resolve_official_evaluation(),self.original.resolve())
            self.assertEqual(pixels.resolve_official_evaluation(pixels.BUNDLED_EVALUATION),pixels.BUNDLED_EVALUATION.resolve())
        with patch.dict(os.environ,{'TEXTEDITBENCH_EVALUATION_DIR':str(self.root/'bad')}):
            with self.assertRaises(FileNotFoundError):pixels.resolve_official_evaluation()

    def test_prompt_fallback_has_identical_snapshot_bytes(self):
        for item in self.prompts.values():
            moved=dict(item,source_file=str(self.root/'missing'/Path(item['source_file']).name))
            resolved=judge.resolve_prompt_source(moved)
            self.assertEqual(resolved,judge.BUNDLED_EVALUATION/Path(item['source_file']).name)
            self.assertEqual(judge.digest(resolved.read_bytes()),item['source_sha256'])

    def test_existing_changed_prompt_source_is_rejected_not_hidden_by_fallback(self):
        item=next(iter(self.prompts.values()))
        changed=self.root/Path(item['source_file']).name;changed.write_text('changed official source')
        with self.assertRaisesRegex(judge.JudgeError,'SHA-256 differs'):
            judge.resolve_prompt_source(dict(item,source_file=str(changed)))

    def test_existing_local_prompt_source_identifier_is_unchanged(self):
        for item in self.prompts.values():
            if Path(item['source_file']).is_file():
                self.assertEqual(judge.resolve_prompt_source(item),Path(item['source_file']).resolve())


if __name__=='__main__':unittest.main()
