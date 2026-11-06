"""Diagnostic-only alpha and immutable RGB/native-ROI provenance contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from output_adapter import (
    ADAPTER_VERSION, instruction_sha256, load_prediction, prepare_rgb_record,
    sha256_file,
)


class NativeAlphaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.folder=self.root/'cases/a';self.folder.mkdir(parents=True)
        self.rgb=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
        self.source=self.folder/'source.png';Image.fromarray(self.rgb).save(self.source)
        self.task={'id':'a','image':'cases/a/source.png','instruction':"Delete 'a'."}
        self.manifest=self.root/'generation_manifest.json'

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self,*,padding=True,padding_alpha=255,native_alpha=255):
        side=64 if padding else 32
        pixels=np.pad(self.rgb,((0,side-32),(0,side-32),(0,0)),mode='edge')
        alpha=np.full((side,side),padding_alpha,np.uint8)
        alpha[:32,:32]=native_alpha
        raw=self.root/'raw.png';Image.fromarray(np.dstack([pixels,alpha])).save(raw)
        meta={'id':'a','source_size':[32,32],'canvas_size':[side,side],
              'evaluation_roi':[0,0,32,32],
              'padding':{'left':0,'top':0,'right':side-32,'bottom':side-32,
                         'mode':'edge' if padding else 'none'},
              'source_sha256':sha256_file(self.source),
              'instruction_sha256':instruction_sha256(self.task),
              'protocol_sha256':'a'*64}
        return raw,meta

    def load(self,record):
        return load_prediction(self.task,self.root,record=record,manifest_path=self.manifest)

    def test_transparent_padding_is_accepted_and_only_native_roi_is_returned(self):
        raw,meta=self.fixture(padding_alpha=0)
        before=raw.read_bytes();export=self.root/'rgb.png'
        record=prepare_rgb_record(raw,export,meta)
        self.assertEqual(record['status'],'complete')
        self.assertEqual(record['adapter_version'],ADAPTER_VERSION)
        self.assertEqual(record['alpha']['scope'],'native_evaluation_roi')
        self.assertEqual(record['alpha']['minimum'],255)
        self.assertEqual(record['alpha_canvas']['minimum'],0)
        self.assertEqual(record['alpha_canvas']['fraction_below_128'],.75)
        output,provenance=self.load(record)
        self.assertTrue(np.array_equal(output,self.rgb))
        self.assertEqual(raw.read_bytes(),before)
        with Image.open(raw) as original,Image.open(export) as rgb:
            self.assertEqual(rgb.size,(64,64))
            self.assertTrue(np.array_equal(np.asarray(rgb),np.asarray(original)[:,:,:3]))
        self.assertEqual(provenance['source_sha256'],sha256_file(self.source))
        self.assertEqual(provenance['instruction_sha256'],instruction_sha256(self.task))
        self.assertEqual(provenance['raw_sha256'],sha256_file(raw))
        self.assertEqual(provenance['rgb_sha256'],sha256_file(export))
        self.assertEqual(provenance['generation_protocol_sha256'],'a'*64)

    def test_fully_transparent_native_pixels_keep_exact_rgb_without_compositing(self):
        raw,meta=self.fixture(native_alpha=0)
        before=raw.read_bytes();export=self.root/'rgb.png'
        record=prepare_rgb_record(raw,export,meta)
        self.assertEqual(record['status'],'complete')
        output,provenance=self.load(record)
        np.testing.assert_array_equal(output,self.rgb)
        self.assertEqual(provenance['alpha_evaluation_roi']['minimum'],0)
        self.assertEqual(provenance['alpha_evaluation_roi']['fraction_below_128'],1)
        self.assertTrue(export.exists());self.assertEqual(raw.read_bytes(),before)

    def test_native_alpha_fraction_is_diagnostic_and_uses_native_denominator(self):
        for count in [1,2,32]:
            raw,meta=self.fixture(padding_alpha=0)
            pixels=np.asarray(Image.open(raw)).copy()
            pixels[0,:count,3]=249
            Image.fromarray(pixels).save(raw)
            record=prepare_rgb_record(raw,self.root/f'export-{count}.png',meta)
            self.assertEqual(record['status'],'complete')
            self.assertEqual(record['alpha']['fraction_below_250'],count/1024)
            np.testing.assert_array_equal(self.load(record)[0],self.rgb)

    def test_no_padding_uses_same_policy(self):
        for alpha in [255,250,249,128,127,0]:
            raw,meta=self.fixture(padding=False,native_alpha=alpha)
            record=prepare_rgb_record(raw,self.root/f'export-{alpha}.png',meta)
            self.assertEqual(record['status'],'complete')
            np.testing.assert_array_equal(self.load(record)[0],self.rgb)
            roi=dict(record['alpha']);roi.pop('scope')
            self.assertEqual(roi,record['alpha_canvas'])

    def test_invalid_padding_cannot_hide_native_transparency(self):
        raw,meta=self.fixture(padding_alpha=0)
        meta['evaluation_roi']=[0,0,31,32]
        with self.assertRaisesRegex(ValueError,'Only recorded native'):
            prepare_rgb_record(raw,self.root/'rgb.png',meta)
        raw,meta=self.fixture(padding_alpha=0)
        meta['padding']['right']=31
        with self.assertRaisesRegex(ValueError,'Inconsistent padding'):
            prepare_rgb_record(raw,self.root/'rgb.png',meta)

    def test_load_recomputes_alpha_statistics_without_rejecting_or_trusting_saved_statistics(self):
        raw,meta=self.fixture(padding_alpha=0)
        record=prepare_rgb_record(raw,self.root/'rgb.png',meta)
        pixels=np.asarray(Image.open(raw)).copy();pixels[0,0,3]=0
        Image.fromarray(pixels).save(raw)
        record['raw_sha256']=sha256_file(raw)
        record['alpha']['minimum']=255
        output,provenance=self.load(record)
        np.testing.assert_array_equal(output,self.rgb)
        self.assertEqual(provenance['alpha_evaluation_roi']['minimum'],0)
        self.assertEqual(provenance['alpha_evaluation_roi']['fraction_below_128'],1/1024)

    def test_alpha_only_raw_tampering_still_fails_hash_validation(self):
        raw,meta=self.fixture()
        record=prepare_rgb_record(raw,self.root/'rgb.png',meta)
        pixels=np.asarray(Image.open(raw)).copy();pixels[0,0,3]=0
        Image.fromarray(pixels).save(raw)
        with self.assertRaisesRegex(ValueError,'Raw output hash mismatch'):
            self.load(record)

    def test_legacy_v1_record_is_read_without_rewriting_or_claiming_new_generation(self):
        raw,meta=self.fixture()
        record=prepare_rgb_record(raw,self.root/'rgb.png',meta)
        record['adapter_version']='raw-rgb-roi-v1'
        record['alpha']=dict(record.pop('alpha_canvas'))
        record['alpha_policy']='Legacy full-canvas alpha policy'
        self.manifest.write_text(json.dumps({'schema_version':1,'records':[record]}))
        before_manifest=self.manifest.read_bytes();before_record=copy.deepcopy(record)
        before_raw=raw.read_bytes();before_rgb=Path(record['rgb_output']).read_bytes()
        _,provenance=self.load(record)
        self.assertEqual(provenance['adapter_version'],'raw-rgb-roi-v1')
        self.assertEqual(provenance['validation_adapter_version'],ADAPTER_VERSION)
        self.assertEqual(record,before_record)
        self.assertEqual(self.manifest.read_bytes(),before_manifest)
        self.assertEqual(raw.read_bytes(),before_raw)
        self.assertEqual(Path(record['rgb_output']).read_bytes(),before_rgb)

    def test_native_rgb_has_uniform_hash_geometry_and_input_provenance(self):
        prediction=self.root/'a.png';Image.fromarray(self.rgb).save(prediction)
        result,provenance=load_prediction(self.task,self.root,predictions_dir=self.root)
        self.assertTrue(np.array_equal(result,self.rgb))
        self.assertEqual(provenance['raw_sha256'],provenance['rgb_sha256'])
        self.assertEqual(provenance['raw_output'],provenance['rgb_output'])
        self.assertEqual(provenance['source_sha256'],sha256_file(self.source))
        self.assertEqual(provenance['instruction_sha256'],instruction_sha256(self.task))
        self.assertEqual(provenance['source_size'],[32,32])
        self.assertEqual(provenance['canvas_size'],[32,32])
        self.assertEqual(provenance['padding']['mode'],'none')
        self.assertIsNone(provenance['generation_protocol_sha256'])

    def test_model_input_hash_is_checked_when_available(self):
        raw,meta=self.fixture()
        model_input=self.root/'model_input.png';Image.fromarray(self.rgb).save(model_input)
        meta.update(model_input=str(model_input),model_input_sha256=sha256_file(model_input))
        record=prepare_rgb_record(raw,self.root/'rgb.png',meta)
        _,provenance=self.load(record)
        self.assertEqual(provenance['model_input_sha256'],sha256_file(model_input))
        self.assertEqual(provenance['model_input'],str(model_input.resolve()))
        Image.new('RGB',(32,32),'red').save(model_input)
        with self.assertRaisesRegex(ValueError,'Model input hash mismatch'):
            self.load(record)

    def test_external_rgba_uses_exact_rgb_channels_without_rewriting_or_alpha_gate(self):
        prediction=self.root/'a.png'
        for alpha in (0,127,255):
            with self.subTest(alpha=alpha):
                pixels=np.dstack([self.rgb,np.full((32,32),alpha,np.uint8)])
                Image.fromarray(pixels).save(prediction);before=prediction.read_bytes()
                result,provenance=load_prediction(self.task,self.root,predictions_dir=self.root)
                self.assertTrue(np.array_equal(result,self.rgb))
                self.assertEqual(prediction.read_bytes(),before)
                self.assertEqual(provenance['mode'],'native_rgba_rgb_channels')
                self.assertEqual(provenance['alpha_evaluation_roi']['minimum'],alpha)
                self.assertEqual(provenance['raw_sha256'],sha256_file(prediction))
                self.assertEqual(provenance['rgb_output'],str(prediction.resolve()))

    def test_external_rgba_still_rejects_unrecorded_resizing(self):
        Image.new('RGBA',(64,32),(20,40,60,0)).save(self.root/'a.png')
        with self.assertRaisesRegex(ValueError,'Dimension mismatch'):
            load_prediction(self.task,self.root,predictions_dir=self.root)


if __name__=='__main__':unittest.main()
