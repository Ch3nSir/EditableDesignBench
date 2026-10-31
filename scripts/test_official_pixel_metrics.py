"""Differential tests against the unmodified local TextEditBench sources."""
import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import official_pixel_metrics as metrics


def direct_module(stem):
    path = metrics.OFFICIAL_EVALUATION/(stem+'.py')
    spec = importlib.util.spec_from_file_location('parity_direct_'+stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        cls.psnr_source = direct_module('mse_psnr')
        cls.perceptual_source = direct_module('lipis_ssim')

    @staticmethod
    def images():
        random = np.random.default_rng(1923)
        source = random.integers(0,256,(128,160,3),dtype=np.uint8)
        # Distinct corners and texture provide enough features for SIFT.
        cv2.rectangle(source,(9,7),(73,61),(240,20,170),2)
        cv2.putText(source,'TEXT',(17,96),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,240,30),1)
        prediction = cv2.warpAffine(source,np.array([[1,0,2],[0,1,-1]],np.float32),
                                    (160,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        prediction[25:38,35:62] = (20,70,210)
        m1 = np.zeros((128,160),np.uint8);m1[22:41,30:64] = 255
        m2 = np.zeros_like(m1);m2[19:39,33:68] = 255
        return source,prediction,m1,m2

    def direct_scores(self,source,prediction,m1,m2):
        union = cv2.bitwise_or(m1,m2)
        s = cv2.cvtColor(source,cv2.COLOR_RGB2BGR)
        p = cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR)
        cv2.setRNGSeed(421)
        aligned = self.psnr_source.align_images(s,p)
        self.assertIsNotNone(aligned)
        mse,psnr = self.psnr_source.calculate_masked_metrics(s,aligned,union)
        inverse = (union<128).astype(np.uint8)*255
        calc = self.perceptual_source.MetricsCalculator('cpu')
        return {'mse':round(mse,4),'psnr':round(psnr,4),
                'ssim':calc.calculate_ssim(prediction,source,inverse,inverse),
                'lpips':calc.calculate_lpips(prediction,source,inverse,inverse)}

    def test_numerical_parity_all_four_metrics(self):
        args = self.images()
        expected = self.direct_scores(*args)
        cv2.setRNGSeed(421)
        actual = metrics.score_case(*args,device='cpu')
        for name in expected:
            self.assertEqual(actual[name],expected[name],name)
        self.assertEqual(actual['metric_status'],{name:'ok' for name in expected})
        self.assertEqual(actual['metric_errors'],{})

    def test_no_identity_ssim_shortcut_and_infinite_psnr_preserved(self):
        source,_,m1,m2 = self.images()
        expected = self.direct_scores(source,source,m1,m2)
        cv2.setRNGSeed(421)
        actual = metrics.score_case(source,source,m1,m2,device='cpu')
        self.assertEqual(actual['ssim'],expected['ssim'])
        self.assertEqual(actual['lpips'],expected['lpips'])
        self.assertEqual(actual['mse'],0)
        self.assertTrue(math.isinf(actual['psnr']))

    def test_aggregation_is_mean_case_psnr_not_psnr_mean_mse(self):
        rows = [{'mse':100.,'psnr':round(10*np.log10(65025/100),4),'ssim':.8,'lpips':.2},
                {'mse':1000.,'psnr':round(10*np.log10(65025/1000),4),'ssim':.6,'lpips':.4}]
        result = metrics.summarize(rows)
        self.assertEqual(result['avg_mse'],550)
        self.assertEqual(result['avg_psnr'],np.mean([r['psnr'] for r in rows]))
        self.assertNotEqual(result['avg_psnr'],10*np.log10(65025/550))
        self.assertEqual(result['avg_ssim'],.7)
        rows[0]['psnr'] = 'inf'
        self.assertTrue(math.isinf(metrics.summarize(rows)['avg_psnr']))

    def test_alignment_failure_does_not_suppress_ssim_or_lpips(self):
        blank = np.full((64,64,3),255,np.uint8)
        mask = np.zeros((64,64),np.uint8)
        result = metrics.score_case(blank,blank,mask,mask)
        for key in ('mse','psnr'):
            self.assertIsNone(result[key])
            self.assertEqual(result['metric_status'][key],'error')
            self.assertIn('alignment failed',result['metric_errors'][key])
        for key in ('ssim','lpips'):
            self.assertTrue(math.isfinite(result[key]))
            self.assertEqual(result['metric_status'][key],'ok')

    def test_perceptual_initialization_failures_are_independent(self):
        module = metrics._load_official(str(metrics.OFFICIAL_EVALUATION.resolve()),'lipis_ssim')
        for key,constructor in [('ssim','StructuralSimilarityIndexMeasure'),
                                ('lpips','LearnedPerceptualImagePatchSimilarity')]:
            with self.subTest(key=key),patch.object(module,constructor,side_effect=RuntimeError('fixture initialization failure')):
                result = metrics.score_case(*self.images())
            self.assertIsNone(result[key])
            self.assertEqual(result['metric_status'][key],'error')
            self.assertIn('fixture initialization failure',result['metric_errors'][key])
            for other in set(metrics.METRICS)-{key}:
                self.assertEqual(result['metric_status'][other],'ok',other)

    def test_perceptual_calculation_failures_are_independent(self):
        module = metrics._load_official(str(metrics.OFFICIAL_EVALUATION.resolve()),'lipis_ssim')
        for key in ('ssim','lpips'):
            with self.subTest(key=key),patch.object(module.MetricsCalculator,'calculate_'+key,
                                                   side_effect=RuntimeError('fixture calculation failure')):
                result = metrics.score_case(*self.images())
            self.assertIsNone(result[key])
            self.assertEqual(result['metric_status'][key],'error')
            self.assertIn('fixture calculation failure',result['metric_errors'][key])
            for other in set(metrics.METRICS)-{key}:
                self.assertEqual(result['metric_status'][other],'ok',other)

    def test_metric_paths_have_independent_import_failures(self):
        load = metrics._load_official
        for stem,affected in [('mse_psnr',{'mse','psnr'}),('lipis_ssim',{'ssim','lpips'})]:
            def loader(folder,requested):
                if requested==stem:raise ImportError('fixture missing optional dependency')
                return load(folder,requested)
            with self.subTest(stem=stem),patch.object(metrics,'_load_official',side_effect=loader):
                result = metrics.score_case(*self.images())
            for key in metrics.METRICS:
                self.assertEqual(result['metric_status'][key],'error' if key in affected else 'ok')
                if key in affected:self.assertIn('missing optional dependency',result['metric_errors'][key])

    def test_nonfinite_values_fail_only_the_affected_metric(self):
        mse_module = metrics._load_official(str(metrics.OFFICIAL_EVALUATION.resolve()),'mse_psnr')
        for pair,bad_key in [((float('nan'),20.),'mse'),((100.,float('nan')),'psnr')]:
            with self.subTest(key=bad_key),patch.object(mse_module,'calculate_masked_metrics',return_value=pair):
                result = metrics.score_case(*self.images())
            self.assertIsNone(result[bad_key])
            self.assertEqual(result['metric_status'][bad_key],'error')
            self.assertIn('non-finite',result['metric_errors'][bad_key])
            for other in set(metrics.METRICS)-{bad_key}:
                self.assertEqual(result['metric_status'][other],'ok',other)
        module = metrics._load_official(str(metrics.OFFICIAL_EVALUATION.resolve()),'lipis_ssim')
        with patch.object(module.MetricsCalculator,'calculate_ssim',return_value=float('nan')):
            result = metrics.score_case(*self.images())
        self.assertIsNone(result['ssim'])
        self.assertIn('non-finite',result['metric_errors']['ssim'])
        self.assertEqual(result['metric_status']['lpips'],'ok')

    def test_selected_metric_skips_unrequested_paths_and_preserves_numerical_value(self):
        args=self.images();expected=self.direct_scores(*args)
        module=metrics._load_official(str(metrics.OFFICIAL_EVALUATION.resolve()),'lipis_ssim')
        with patch.object(module,'LearnedPerceptualImagePatchSimilarity',side_effect=AssertionError('LPIPS must be skipped')):
            result=metrics.score_case(*args,metrics=['ssim'])
        self.assertEqual(result['ssim'],expected['ssim'])
        self.assertEqual(result['metric_status'],{'mse':'skipped','psnr':'skipped','ssim':'ok','lpips':'skipped'})
        with patch.object(metrics,'_load_official',side_effect=lambda folder,stem:self.psnr_source if stem=='mse_psnr' else self.fail('Perceptual path must be skipped')):
            cv2.setRNGSeed(421);result=metrics.score_case(*args,metrics=['mse'])
        self.assertEqual(result['mse'],expected['mse'])
        self.assertEqual(result['metric_status']['psnr'],'skipped')

    def test_input_arrays_are_not_mutated(self):
        args = self.images();before = [a.copy() for a in args]
        cv2.setRNGSeed(421);metrics.score_case(*args)
        for actual,expected in zip(args,before):self.assertTrue(np.array_equal(actual,expected))

    def test_provenance_and_strict_native_geometry(self):
        p = metrics.protocol_info()
        self.assertEqual(set(p['sources']),{'mse_psnr.py','lipis_ssim.py'})
        self.assertTrue(all(Path(x['path']).is_file() and len(x['sha256'])==64 for x in p['sources'].values()))
        s,pred,m1,m2 = self.images()
        with self.assertRaisesRegex(ValueError,'identical native RGB'):
            metrics.score_case(s,pred[:-1],m1,m2)


if __name__=='__main__':unittest.main()
