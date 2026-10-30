"""Real temporary artifacts must match every stage of smoke acceptance.

Only numerical metric work is mocked. Images, adapters, pixel report structure,
MLLM dry-run packaging, preview rendering and final acceptance use real code.
No model, GPU, API request or production dataset is used.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate
import mllm_evaluate
from output_adapter import instruction_sha256, prepare_rgb_record, sha256_file
import preview_run
import smoke_report
from evaluation_binding import canonical_hash


class ReportBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.dataset=self.root/'dataset';self.folder=self.dataset/'cases/a'
        self.folder.mkdir(parents=True);self.run=self.root/'run';self.run.mkdir()
        self.pixels=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
        self.source=self.folder/'source.png';Image.fromarray(self.pixels).save(self.source)
        Image.fromarray(self.pixels).save(self.folder/'target.png')
        mask=np.zeros((64,64),np.uint8);mask[5:20,5:25]=255
        for name in ('mask.png','target_region_mask.png'):Image.fromarray(mask).save(self.folder/name)
        box={'x':5,'y':5,'width':20,'height':15}
        self.annotation={'source_geometry':{'box':box},'target_geometry':{'box':dict(box)}}
        (self.folder/'annotation.json').write_text(json.dumps(self.annotation))
        self.task={'id':'a','source_id':'source-a','image':'cases/a/source.png',
                   'instruction':"Delete 'a'.",'operation':'delete','domain':'poster',
                   'output_filename':'a.png'}
        (self.dataset/'tasks.jsonl').write_text(json.dumps(self.task)+'\n')
        (self.dataset/'summary.json').write_text('{"version":"fixture"}\n')
        self.summary_before=(self.dataset/'summary.json').read_bytes()
        self.manifest=self.run/'generation_manifest.json'
        self.protocol={'case_ids':['a']}
        self.protocol['protocol_sha256']=canonical_hash(self.protocol)
        (self.run/'protocol.json').write_text(json.dumps(self.protocol))
        self.record=self.make_record('attempt_001',self.pixels)
        self.write_manifest(self.record)
        self.build_reports()

    def tearDown(self):
        self.assertEqual((self.dataset/'summary.json').read_bytes(),self.summary_before)
        self.temp.cleanup()

    def make_record(self,attempt,pixels):
        directory=self.run/attempt;directory.mkdir()
        raw=directory/'raw.png';rgb=directory/'prediction.png'
        Image.fromarray(np.dstack([pixels,np.full((64,64),255,np.uint8)])).save(raw)
        metadata={'id':'a','source_size':[64,64],'canvas_size':[64,64],
                  'evaluation_roi':[0,0,64,64],
                  'padding':{'left':0,'top':0,'right':0,'bottom':0,'mode':'none'},
                  'source_sha256':sha256_file(self.source),
                  'instruction_sha256':instruction_sha256(self.task),
                  'protocol_sha256':self.protocol['protocol_sha256']}
        return prepare_rgb_record(raw,rgb,metadata)

    def write_manifest(self,record):
        self.manifest.write_text(json.dumps({'schema_version':1,'status':'complete',
            'protocol_sha256':self.protocol['protocol_sha256'],'expected_ids':['a'],'records':[record]}))

    @staticmethod
    def scores(*args,**kwargs):
        values={'mse':10.,'psnr':38.1308,'ssim':.99,'lpips':.01}
        return {**values,'metric_status':dict.fromkeys(values,'ok'),'metric_errors':{}}

    def build_reports(self):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),\
             patch.object(evaluate,'score_case',side_effect=self.scores),\
             patch.object(evaluate,'_configure_runtime',return_value=[]):
            code=evaluate.main(['--dataset-root',str(self.dataset),'--generation-manifest',str(self.manifest),
                                '--output',str(self.run/'pixel_scores.json')])
        self.assertEqual(code,0)
        args=mllm_evaluate.parser().parse_args([
            '--dataset',str(self.dataset),'--generation-manifest',str(self.manifest),
            '--output',str(self.run/'mllm_pending.json'),'--dry-run'])
        with patch.dict(os.environ,{},clear=True):
            judge,code=mllm_evaluate.evaluate(args,transport=lambda *a:self.fail('Real API transport called'))
        self.assertEqual(code,0);self.assertEqual(judge['api_calls_made'],0)
        arguments=['preview_run.py','--generation-manifest',str(self.manifest),
                   '--output',str(self.run/'comparison.png'),'--dataset-root',str(self.dataset)]
        with patch.object(sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preview_run.main(),0)

    def collect(self):
        arguments=['smoke_report.py','--run-dir',str(self.run),'--dataset-root',str(self.dataset),
                   '--generation-exit','0','--pixel-exit','0','--mllm-exit','0','--preview-exit','0',
                   '--no-update-dataset']
        with patch.object(sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):
            code=smoke_report.main()
        return code,json.loads((self.run/'smoke_report.json').read_text())

    def assert_rejected(self,*,stage=None,reason=None):
        code,report=self.collect()
        self.assertEqual(code,1);self.assertFalse(report['generation_and_pixel_smoke_passed'])
        if stage:self.assertFalse(report['stages'][stage]['passed'])
        if reason:self.assertTrue(any(reason in msg for msg in report['validation_errors']),report)
        return report

    def test_current_images_and_reports_pass_all_four_stages(self):
        code,report=self.collect()
        self.assertEqual(code,0);self.assertTrue(report['generation_and_pixel_smoke_passed'])
        self.assertTrue(all(stage['passed'] for stage in report['stages'].values()))
        self.assertFalse(report['full_semantic_evaluation_complete'])
        self.assertEqual(report['mllm_api_calls'],0)

    def test_malformed_stage_report_replaces_old_pass_with_failure(self):
        self.assertEqual(self.collect()[0],0)
        (self.run/'pixel_scores.json').write_text('{truncated')
        self.assert_rejected(stage='pixel_scoring',reason='pixel_scores.json')

    def test_generation_protocol_body_cannot_change_behind_old_fingerprint(self):
        changed=dict(self.protocol,steps=999)
        (self.run/'protocol.json').write_text(json.dumps(changed))
        self.assert_rejected(stage='generation',reason='protocol body')

    def test_official_source_metadata_cannot_be_omitted(self):
        path=self.run/'pixel_scores.json';pixel=json.loads(path.read_text())
        pixel['protocol']['sources']={};path.write_text(json.dumps(pixel))
        self.assert_rejected(stage='pixel_scoring',reason='Missing official pixel source')

    def test_same_ids_with_unmatched_generated_hashes_do_not_pass(self):
        record=copy.deepcopy(self.record)
        record.update(raw_sha256='2'*64,rgb_sha256='2'*64)
        self.write_manifest(record)
        self.assert_rejected(stage='generation',reason='hash mismatch')

    def test_valid_new_attempt_cannot_reuse_old_pixel_or_judge_reports(self):
        pixels=self.pixels.copy();pixels[30:35,30:35]=[255,20,10]
        replacement=self.make_record('attempt_002',pixels)
        self.write_manifest(replacement)
        report=self.assert_rejected(stage='pixel_scoring',reason='provenance changed')
        self.assertTrue(report['stages']['generation']['passed'])
        self.assertFalse(report['stages']['mllm_input_validation']['passed'])
        self.assertFalse(report['stages']['comparison_preview']['passed'])

    def test_mask_changes_invalidate_pixel_input_binding(self):
        path=self.folder/'target_region_mask.png'
        with Image.open(path) as im:mask=np.asarray(im).copy()
        mask[25:30,25:30]=255;Image.fromarray(mask).save(path)
        report=self.assert_rejected(stage='pixel_scoring',reason='mask binding changed')
        self.assertTrue(report['stages']['generation']['passed'])

    def test_source_changes_reject_generation_and_all_dependent_stages(self):
        pixels=self.pixels.copy();pixels[0,0]=[0,0,0];Image.fromarray(pixels).save(self.source)
        self.assert_rejected(stage='generation',reason='Source hash')

    def test_instruction_changes_reject_generation_and_stale_scoring(self):
        changed=dict(self.task,instruction="Delete 'b'.")
        (self.dataset/'tasks.jsonl').write_text(json.dumps(changed)+'\n')
        self.assert_rejected(stage='generation',reason='Instruction changed')

    def test_modified_preview_png_is_rejected(self):
        Image.new('RGB',(64,64),'red').save(self.run/'comparison.png')
        report=self.assert_rejected(stage='comparison_preview',reason='Preview image changed')
        self.assertTrue(report['stages']['pixel_scoring']['passed'])

    def test_reference_and_annotation_changes_invalidate_preview(self):
        reference=self.folder/'target.png';reference_before=reference.read_bytes()
        Image.new('RGB',(64,64),'red').save(reference)
        self.assert_rejected(stage='comparison_preview',reason='Preview inputs changed')
        reference.write_bytes(reference_before)
        changed=copy.deepcopy(self.annotation);changed['target_geometry']['box']['x']=15
        (self.folder/'annotation.json').write_text(json.dumps(changed))
        self.assert_rejected(stage='comparison_preview',reason='Preview inputs changed')

    def test_changed_evaluator_hash_invalidates_pixel_report(self):
        path=self.run/'pixel_scores.json';pixel=json.loads(path.read_text())
        pixel['protocol']['evaluator_sources']['evaluate.py']['sha256']='1'*64
        path.write_text(json.dumps(pixel))
        self.assert_rejected(stage='pixel_scoring',reason='Pixel evaluator source changed')

    def test_deleted_output_cannot_be_accepted_from_cached_report(self):
        Path(self.record['raw_output']).unlink()
        self.assert_rejected(stage='generation')

    def test_pixel_gate_alone_rechecks_live_masks(self):
        pixel=json.loads((self.run/'pixel_scores.json').read_text())
        self.assertTrue(smoke_report.pixel_report_complete(pixel,['a']))
        Image.new('L',(64,64),0).save(self.folder/'mask.png')
        errors=[]
        self.assertFalse(smoke_report.pixel_report_complete(pixel,['a'],errors=errors))
        self.assertTrue(any('mask binding changed' in msg for msg in errors),errors)


if __name__=='__main__':unittest.main()
