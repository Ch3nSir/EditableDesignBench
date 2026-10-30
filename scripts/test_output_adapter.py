"""Focused contract tests; all fixtures are temporary, no real model/API needed."""
import contextlib
import builtins
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate
import smoke_report
from smoke_report import pixel_report_complete
from output_adapter import instruction_sha256, load_prediction, prepare_rgb_record, sha256_file


class AdapterContract(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.dataset=self.root/'dataset';self.folder=self.dataset/'cases/a';self.folder.mkdir(parents=True)
        self.source=self.folder/'source.png';self.rgb=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
        Image.fromarray(self.rgb).save(self.source);Image.fromarray(self.rgb).save(self.folder/'target.png')
        mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255;Image.fromarray(mask).save(self.folder/'eval_mask.png')
        Image.fromarray(mask).save(self.folder/'mask.png');Image.fromarray(mask).save(self.folder/'target_region_mask.png')
        self.task={'id':'a','image':'cases/a/source.png','instruction':"Delete 'a'.",'operation':'delete','domain':'poster','output_filename':'a.png'}
        (self.dataset/'tasks.jsonl').write_text(json.dumps(self.task)+'\n')
        self.predictions=self.root/'predictions';self.predictions.mkdir()

    def tearDown(self):self.temporary.cleanup()

    def metadata(self):
        return {'id':'a','source_size':[32,32],'canvas_size':[64,64],'evaluation_roi':[0,0,32,32],
            'padding':{'left':0,'top':0,'right':32,'bottom':32,'mode':'edge'},
            'source_sha256':sha256_file(self.source),'instruction_sha256':instruction_sha256(self.task),
            'seed':42,'steps':1,'model':'test-fixture'}

    def raw(self,alpha=255):
        padded=np.pad(self.rgb,((0,32),(0,32),(0,0)),mode='edge')
        rgba=np.dstack([padded,np.full((64,64),alpha,np.uint8)])
        path=self.root/'raw.png';Image.fromarray(rgba).save(path);return path

    @staticmethod
    def metric_result(*,mse=0.,psnr=float('inf'),ssim=1.,lpips=0.,errors=None):
        values={'mse':mse,'psnr':psnr,'ssim':ssim,'lpips':lpips}
        errors=errors or {}
        return {**values,'metric_status':{key:'error' if key in errors else 'ok' for key in values},
                'metric_errors':dict(errors)}

    def add_task(self,cid,*,operation='move',domain='chart'):
        folder=self.dataset/'cases'/cid;folder.mkdir()
        for name in ('source.png','target.png','mask.png','target_region_mask.png','eval_mask.png'):
            (folder/name).write_bytes((self.folder/name).read_bytes())
        task=dict(self.task,id=cid,image=f'cases/{cid}/source.png',operation=operation,
                  domain=domain,output_filename=f'{cid}.png')
        with (self.dataset/'tasks.jsonl').open('a') as stream:stream.write(json.dumps(task)+'\n')
        Image.fromarray(self.rgb).save(self.predictions/f'{cid}.png')
        return task

    def run_evaluator(self,*arguments,score_outputs=None,real_runtime=False):
        output=self.root/'nested/scores.json'
        # These tests cover the output/CLI contract. Official numerical parity
        # has independent real SIFT/SSIM/LPIPS tests in test_official_pixel_metrics.
        def fixture_scores(*args,**kwargs):return self.metric_result()
        runtime=contextlib.nullcontext() if real_runtime else patch.object(evaluate,'_configure_runtime',return_value=[])
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),patch.object(evaluate,'score_case',side_effect=score_outputs if score_outputs is not None else fixture_scores),runtime:
            code=evaluate.main(['--dataset-root',str(self.dataset),'--output',str(output),*arguments])
        return code,json.loads(output.read_text())

    def test_native_identity(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        code,result=self.run_evaluator('--predictions',str(self.predictions))
        self.assertEqual(code,0);self.assertTrue(result['complete']);self.assertEqual(result['valid'],1)
        self.assertEqual(result['summary']['metrics']['avg_mse'],0)
        self.assertEqual(result['summary']['metrics']['avg_psnr'],'inf')
        self.assertEqual(result['protocol']['version'],'texteditbench-official-pixels-v2')
        self.assertTrue(all(v['complete'] for v in result['summary']['metric_coverage'].values()))

    def test_missing_prediction_fails_without_subset_average(self):
        code,result=self.run_evaluator('--predictions',str(self.predictions))
        self.assertEqual(code,1);self.assertEqual(result['failed'],1);self.assertEqual(result['coverage'],0)
        self.assertEqual(result['summary']['metrics'],dict.fromkeys(('avg_mse','avg_psnr','avg_ssim','avg_lpips')))
        row=result['results'][0]
        self.assertEqual(row['failure_stage'],'input')
        self.assertEqual(row['metric_status'],dict.fromkeys(('mse','psnr','ssim','lpips'),'error'))
        self.assertEqual(set(row['metric_errors']),{'mse','psnr','ssim','lpips'})
        for coverage in result['summary']['metric_coverage'].values():
            self.assertEqual(coverage,{'expected':1,'valid':0,'failed':1,'pending':0,'coverage':0.,'complete':False})

    def test_unknown_and_duplicate_ids_fail(self):
        for ids in ['unknown','a,a','a,']:
            code,result=self.run_evaluator('--predictions',str(self.predictions),'--ids',ids)
            self.assertEqual(code,2);self.assertEqual(result['status'],'invalid_request')

    def test_opaque_rgba_needs_explicit_export(self):
        Image.fromarray(np.dstack([self.rgb,np.full((32,32),255,np.uint8)])).save(self.predictions/'a.png')
        code,result=self.run_evaluator('--predictions',str(self.predictions))
        self.assertEqual(code,1);self.assertIn('Expected RGB',result['results'][0]['error'])

    def test_padded_rgba_export_preserves_raw_and_evaluates_native_roi(self):
        raw=self.raw();before=raw.read_bytes();export=self.root/'rgb.png'
        record=prepare_rgb_record(raw,export,self.metadata())
        self.assertEqual(record['status'],'complete');self.assertEqual(raw.read_bytes(),before)
        self.assertEqual(record['seed'],42)
        manifest=self.root/'generation.json';manifest.write_text(json.dumps({'schema_version':1,'records':[record]}))
        pixels,_=load_prediction(self.task,self.dataset,record=record,manifest_path=manifest)
        self.assertTrue(np.array_equal(pixels,self.rgb))
        with Image.open(raw) as raw_image:self.assertEqual(raw_image.size,(64,64))
        code,result=self.run_evaluator('--generation-manifest',str(manifest))
        self.assertEqual(code,0);self.assertEqual(result['results'][0]['mse'],0.)

    def test_reject_substantive_alpha_and_canvas_mismatch(self):
        raw=self.raw(alpha=0);before=raw.read_bytes();export=self.root/'rgb.png'
        record=prepare_rgb_record(raw,export,self.metadata())
        self.assertEqual(record['status'],'rejected_alpha');self.assertFalse(export.exists());self.assertEqual(raw.read_bytes(),before)
        meta=self.metadata();meta['canvas_size']=[32,32];meta['padding'].update(right=0,bottom=0)
        with self.assertRaisesRegex(ValueError,'Actual output'):prepare_rgb_record(raw,export,meta)

    def test_alpha_threshold_boundary(self):
        for count,level,expected in [(1,249,'complete'),(2,249,'rejected_alpha'),(1,127,'rejected_alpha')]:
            raw=self.raw();pixels=np.asarray(Image.open(raw)).copy();pixels.reshape(-1,4)[:count,3]=level
            Image.fromarray(pixels).save(raw)
            record=prepare_rgb_record(raw,self.root/f'rgb-{count}-{level}.png',self.metadata())
            self.assertEqual(record['status'],expected)

    def test_stale_instruction_and_hash_tampering_fail(self):
        raw=self.raw();record=prepare_rgb_record(raw,self.root/'rgb.png',self.metadata());manifest=self.root/'m.json'
        task=dict(self.task,instruction='Different instruction')
        with self.assertRaisesRegex(ValueError,'Instruction changed'):load_prediction(task,self.dataset,record=record,manifest_path=manifest)
        record['raw_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'):load_prediction(self.task,self.dataset,record=record,manifest_path=manifest)

    def test_official_case_means_and_incomplete_groups_suppress_scores(self):
        rows=[dict(self.metric_result(mse=100.,psnr=28.1308,ssim=.8,lpips=.1),status='ok'),
              dict(self.metric_result(mse=1000.,psnr=18.1308,ssim=.6,lpips=.3),status='ok')]
        summary=evaluate.summarize(rows)
        self.assertEqual(summary['metrics']['avg_mse'],550.)
        self.assertEqual(summary['metrics']['avg_psnr'],23.1308)
        self.assertEqual(summary['metrics']['avg_ssim'],.7)
        error=dict(self.metric_result(mse=None,psnr=None,ssim=None,lpips=None,
                   errors=dict.fromkeys(('mse','psnr','ssim','lpips'),'Missing output')),status='error')
        self.assertTrue(all(v is None for v in evaluate.summarize([rows[0],error])['metrics'].values()))

    def test_alignment_failure_keeps_perceptual_scores(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        partial=self.metric_result(mse=None,psnr=None,ssim=.8,lpips=.1,
                  errors={'mse':'Official image alignment failed','psnr':'Official image alignment failed'})
        code,result=self.run_evaluator('--predictions',str(self.predictions),score_outputs=[partial])
        self.assertEqual(code,1);self.assertFalse(result['complete'])
        self.assertEqual((result['valid'],result['failed'],result['partial']),(0,1,1))
        row=result['results'][0];self.assertEqual(row['status'],'partial')
        self.assertEqual(row['failure_stage'],'metrics')
        self.assertIsNone(row['mse']);self.assertIsNone(row['psnr'])
        self.assertEqual(result['summary']['metrics'],{'avg_mse':None,'avg_psnr':None,'avg_ssim':.8,'avg_lpips':.1})
        for key in ('ssim','lpips'):
            self.assertTrue(result['summary']['metric_coverage'][key]['complete'])
            self.assertNotIn(key,row['metric_errors'])
        for key in ('mse','psnr'):
            self.assertEqual(result['summary']['metric_coverage'][key]['failed'],1)
            self.assertIn('alignment failed',row['metric_errors'][key])

    def test_ssim_failure_does_not_hide_lpips_or_aligned_metrics(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        partial=self.metric_result(ssim=None,lpips=.125,errors={'ssim':'SSIM computation failed'})
        code,result=self.run_evaluator('--predictions',str(self.predictions),score_outputs=[partial])
        self.assertEqual(code,1)
        self.assertEqual(result['summary']['metrics'],{'avg_mse':0.,'avg_psnr':'inf','avg_ssim':None,'avg_lpips':.125})
        self.assertEqual(result['summary']['metric_coverage']['ssim']['failed'],1)
        self.assertTrue(result['summary']['metric_coverage']['lpips']['complete'])

    def test_cpu_torch_import_failure_preserves_mse_psnr_and_warning(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        real_import=builtins.__import__
        def without_torch(name,*args,**kwargs):
            if name=='torch' or name.startswith('torch.'):
                raise ModuleNotFoundError('synthetic torch unavailable')
            return real_import(name,*args,**kwargs)
        partial=self.metric_result(mse=100.,psnr=28.1308,ssim=None,lpips=None,
                  errors={'ssim':'ModuleNotFoundError: synthetic torch unavailable',
                          'lpips':'ModuleNotFoundError: synthetic torch unavailable'})
        with patch.object(builtins,'__import__',side_effect=without_torch):
            code,result=self.run_evaluator('--predictions',str(self.predictions),'--device','cpu',
                                          score_outputs=[partial],real_runtime=True)
        self.assertEqual(code,1);self.assertEqual(result['results'][0]['status'],'partial')
        self.assertEqual(result['summary']['metrics'],{'avg_mse':100.,'avg_psnr':28.1308,'avg_ssim':None,'avg_lpips':None})
        self.assertTrue(result['summary']['metric_coverage']['mse']['complete'])
        self.assertTrue(result['summary']['metric_coverage']['psnr']['complete'])
        self.assertTrue(any('synthetic torch unavailable' in message
                            for message in result['protocol']['runtime_warnings']))
        self.assertEqual(result['results'][0]['failure_stage'],'metrics')

    def test_explicit_cuda_without_gpu_fails_without_cpu_fallback(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        with patch('torch.cuda.is_available',return_value=False):
            code,result=self.run_evaluator('--predictions',str(self.predictions),'--device','cuda',
                                          score_outputs=[],real_runtime=True)
        self.assertEqual(code,1);row=result['results'][0]
        self.assertEqual(row['failure_stage'],'runtime')
        self.assertIn('without an allocated visible GPU',row['error'])
        self.assertTrue(all(value is None for value in result['summary']['metrics'].values()))
        self.assertEqual(set(row['metric_errors']),{'mse','psnr','ssim','lpips'})

    def test_metric_denominators_and_operation_domain_groups(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png');self.add_task('b')
        partial=self.metric_result(mse=None,psnr=None,ssim=.7,lpips=.1,
                  errors={'mse':'Alignment failed','psnr':'Alignment failed'})
        good=self.metric_result(mse=100.,psnr=28.1308,ssim=.9,lpips=.3)
        code,result=self.run_evaluator('--predictions',str(self.predictions),score_outputs=[partial,good])
        self.assertEqual(code,1)
        self.assertEqual((result['expected'],result['valid'],result['failed'],result['partial']),(2,1,1,1))
        summary=result['summary'];self.assertIsNone(summary['metrics']['avg_mse'])
        self.assertEqual(summary['metrics']['avg_ssim'],.8)
        self.assertEqual(summary['metric_coverage']['mse'],{'expected':2,'valid':1,'failed':1,'pending':0,'coverage':.5,'complete':False})
        self.assertEqual(summary['metric_coverage']['ssim'],{'expected':2,'valid':2,'failed':0,'pending':0,'coverage':1.,'complete':True})
        for field,left,right in [('by_operation','delete','move'),('by_domain','poster','chart')]:
            self.assertEqual(result[field][left]['expected'],1)
            self.assertEqual(result[field][left]['metrics']['avg_ssim'],.7)
            self.assertIsNone(result[field][left]['metrics']['avg_mse'])
            self.assertTrue(result[field][right]['complete'])
            self.assertEqual(result[field][right]['metrics']['avg_mse'],100.)

    def test_pending_case_remains_in_each_metric_denominator(self):
        good=dict(self.metric_result(),status='ok')
        pending={'status':'pending','metric_status':dict.fromkeys(('mse','psnr','ssim','lpips'),'pending')}
        result=evaluate.summarize([good,pending])
        self.assertEqual((result['expected'],result['valid'],result['failed'],result['pending']),(2,1,0,1))
        self.assertTrue(all(value is None for value in result['metrics'].values()))
        for coverage in result['metric_coverage'].values():
            self.assertEqual(coverage,{'expected':2,'valid':1,'failed':0,'pending':1,'coverage':.5,'complete':False})

    def test_smoke_gate_checks_individual_metric_completion(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        code,complete=self.run_evaluator('--predictions',str(self.predictions))
        self.assertEqual(code,0);self.assertTrue(pixel_report_complete(complete,['a']))
        mutations=[(('summary','metric_coverage','ssim','valid'),0),
                   (('results',0,'metric_status','lpips'),'error'),
                   (('results',0,'metric_errors','ssim'),'Unexpected error'),
                   (('summary','metrics','avg_ssim'),None),
                   (('protocol','version'),'texteditbench-official-pixels-v1')]
        for path,value in mutations:
            with self.subTest(path=path):
                malformed=json.loads(json.dumps(complete));target=malformed
                for key in path[:-1]:target=target[key]
                target[path[-1]]=value
                self.assertFalse(pixel_report_complete(malformed,['a']))

    def test_smoke_summary_records_partial_metric_scores_without_claiming_pass(self):
        Image.fromarray(self.rgb).save(self.predictions/'a.png')
        partial=self.metric_result(mse=None,psnr=None,ssim=.8,lpips=.1,
                  errors={'mse':'Alignment failed','psnr':'Alignment failed'})
        _,pixel=self.run_evaluator('--predictions',str(self.predictions),score_outputs=[partial])
        run=self.root/'run';run.mkdir();summary_path=self.dataset/'summary.json'
        summary_path.write_text('{}\n')
        fixtures={'protocol.json':{'case_ids':['a']},
                  'generation_manifest.json':{'records':[{'id':'a','status':'complete','raw_output':'raw.png'}]},
                  'pixel_scores.json':pixel,
                  'mllm_pending.json':{'status':'not_scored','api_calls_made':0,
                                      'summary':{'errors':0,'expected':1},'results':[{'id':'a'}]}}
        for name,value in fixtures.items():(run/name).write_text(json.dumps(value))
        (run/'comparison.png').write_bytes(self.source.read_bytes())
        arguments=['smoke_report.py','--run-dir',str(run),'--dataset-root',str(self.dataset),'--generation-exit','0',
                   '--pixel-exit','1','--mllm-exit','0','--preview-exit','0']
        def collect():
            with patch.object(sys,'argv',arguments),patch.object(smoke_report,'__file__',str(self.dataset/'scripts/smoke_report.py')),contextlib.redirect_stdout(io.StringIO()):
                code=smoke_report.main()
            return code,json.loads(summary_path.read_text())
        code,summary=collect()
        self.assertEqual(code,1);self.assertTrue(summary['model_scores_computed'])
        self.assertFalse(summary['latest_model_smoke']['passed'])
        self.assertFalse(json.loads((run/'smoke_report.json').read_text())['full_semantic_evaluation_complete'])
        (run/'pixel_scores.json').unlink()
        code,summary=collect()
        self.assertEqual(code,1);self.assertFalse(summary['model_scores_computed'])

    def test_cli_actual_exit_code(self):
        output=self.root/'cli.json'
        result=subprocess.run([sys.executable,str(Path(evaluate.__file__)),'--dataset-root',str(self.dataset),
            '--predictions',str(self.predictions),'--ids','unknown','--output',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('Unknown requested IDs',result.stderr)
        result=subprocess.run([sys.executable,str(Path(evaluate.__file__)),'--dataset-root',str(self.dataset),
            '--predictions',str(self.predictions),'--output',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,1)


if __name__=='__main__':unittest.main()
