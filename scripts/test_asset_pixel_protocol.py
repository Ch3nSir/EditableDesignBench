"""Asset backgrounds use the instruction allowance, independent of one GT."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate
from evaluation_binding import (ASSET_BINDING_VERSION, ASSET_INPUT_PROTOCOL, ASSET_PIXEL_PROTOCOL,
    binding_version, case_binding, dataset_track, preservation_mask_path, resolve_track_root)
from output_adapter import load_prediction


class AssetPixelProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.data=self.root/'dataset';self.folder=self.data/'cases/a';self.folder.mkdir(parents=True)
        self.pred=self.root/'predictions';self.pred.mkdir()
        pixels=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
        Image.fromarray(pixels).save(self.folder/'source.png')
        Image.fromarray(pixels).save(self.pred/'a.png')
        self.task={'id':'a','image':'cases/a/source.png','operation':'move','instruction':'Move the photo slightly right.',
                   'source_id':'source-a','input_protocol':ASSET_INPUT_PROTOCOL,'output_filename':'a.png'}
        (self.data/'tasks.jsonl').write_text(json.dumps(self.task)+'\n')
        self.reference={'id':'a','pixel_preservation_mask':'cases/a/allowed_edit_mask.png',
                        'target_region_mask':'cases/a/target_region_mask.png'}
        self.write_reference()
        allowed=np.zeros((64,64),dtype=np.uint8);allowed[2:42,2:42]=255
        Image.fromarray(allowed).save(self.folder/'allowed_edit_mask.png')
        narrow=np.zeros_like(allowed);narrow[4:10,4:10]=255
        for name in ('mask.png','target_region_mask.png'):Image.fromarray(narrow).save(self.folder/name)
        self.output=self.root/'pixel_scores.json'

    def tearDown(self):self.tmp.cleanup()

    def write_reference(self):
        (self.data/'references.jsonl').write_text(json.dumps(self.reference)+'\n')

    def metric(self,source,prediction,source_mask,target_mask,**kwargs):
        # This location is permitted by the instruction, outside both exact GT
        # masks. The old implementation silently penalized valid alternatives.
        self.assertEqual(int(source_mask[30,30]),255)
        self.assertFalse(target_mask.any())
        values={'mse':0.,'psnr':float('inf'),'ssim':1.,'lpips':0.}
        selected=kwargs.get('metrics',values)
        return {**values,'metric_status':{k:'ok' if k in selected else 'skipped' for k in values},'metric_errors':{}}

    def run_pixel(self):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),\
             patch.object(evaluate,'score_case',side_effect=self.metric) as metric,\
             patch.object(evaluate,'_configure_runtime',return_value=[]):
            code=evaluate.main(['--dataset-root',str(self.data),'--predictions',str(self.pred),'--output',str(self.output)])
        return code,metric.call_count,json.loads(self.output.read_text())

    def test_asset_uses_allowed_region_and_own_protocol(self):
        code,calls,report=self.run_pixel()
        self.assertEqual(code,0);self.assertEqual(calls,1)
        self.assertEqual(report['protocol']['version'],ASSET_PIXEL_PROTOCOL)
        self.assertEqual(report['protocol']['artifact_binding_version'],ASSET_BINDING_VERSION)
        self.assertEqual(report['protocol']['mask_inputs'],['pixel_preservation_mask'])
        self.assertTrue(report['complete'])

    def test_changed_allowed_mask_invalidates_resume(self):
        self.assertEqual(self.run_pixel()[0],0)
        path=self.folder/'allowed_edit_mask.png'
        with Image.open(path) as image:mask=np.asarray(image).copy()
        mask[45,45]=255;Image.fromarray(mask).save(path)
        code,calls,report=self.run_pixel()
        self.assertEqual(code,0);self.assertEqual(calls,1)
        self.assertEqual(report['resume']['reused_metrics'],0)

    def test_exact_gt_masks_do_not_define_asset_background_binding(self):
        _,provenance=load_prediction(self.task,self.data,predictions_dir=self.pred)
        before=case_binding(self.task,self.data,provenance,masks=True)
        Image.new('L',(64,64),255).save(self.folder/'target_region_mask.png')
        self.assertEqual(before,case_binding(self.task,self.data,provenance,masks=True))
        self.assertEqual(binding_version(self.data),ASSET_BINDING_VERSION)

    def test_missing_allowed_reference_never_falls_back_to_gt(self):
        del self.reference['pixel_preservation_mask'];self.write_reference()
        with self.assertRaisesRegex(ValueError,'pixel_preservation_mask'):
            preservation_mask_path(self.task,self.data)

    def test_reference_path_cannot_escape_dataset(self):
        self.reference['pixel_preservation_mask']='../outside.png';self.write_reference()
        Image.new('L',(64,64)).save(self.root/'outside.png')
        with self.assertRaisesRegex(ValueError,'Invalid asset'):
            preservation_mask_path(self.task,self.data)

    def test_mixed_tracks_are_rejected(self):
        text=dict(self.task);text.pop('input_protocol')
        with self.assertRaisesRegex(ValueError,'separately'):
            dataset_track([text,self.task])

    def test_asset_without_explicit_protocol_cannot_use_text_masks(self):
        task=dict(self.task,id='asset_000_a00_move');task.pop('input_protocol')
        with self.assertRaisesRegex(ValueError,'explicitly declare'):
            dataset_track([task])

    def test_asset_cli_track_resolves_nested_or_direct_path(self):
        collection=self.root/'collection';collection.mkdir()
        text=dict(self.task);text.pop('input_protocol')
        (collection/'tasks.jsonl').write_text(json.dumps(text)+'\n')
        assets=collection/'assets';assets.mkdir()
        (assets/'tasks.jsonl').write_text(json.dumps(self.task)+'\n')
        self.assertEqual(resolve_track_root(collection,'asset'),assets)
        self.assertEqual(resolve_track_root(assets,'asset'),assets)
        self.assertEqual(resolve_track_root(collection,'text'),collection)
        with self.assertRaises(ValueError):resolve_track_root(assets,'text')


if __name__=='__main__':unittest.main()
