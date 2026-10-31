"""Per-metric resume with real file bindings and an offline metric stub."""
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


class PixelResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'data';self.predictions=self.root/'predictions'
        self.predictions.mkdir();self.output=self.root/'pixel.json';self.calls=[]
        self.values={'mse':10.,'psnr':38.1308,'ssim':.9,'lpips':.05}
        tasks=[]
        for cid in ('a','b'):
            folder=self.data/'cases'/cid;folder.mkdir(parents=True)
            Image.new('RGB',(64,64),'white').save(folder/'source.png')
            Image.new('RGB',(64,64),'blue').save(self.predictions/(cid+'.png'))
            for name in ('mask.png','target_region_mask.png'):Image.new('L',(64,64),0).save(folder/name)
            tasks.append({'id':cid,'image':f'cases/{cid}/source.png','instruction':f"Delete '{cid}'.",'operation':'delete'})
        (self.data/'tasks.jsonl').write_text(''.join(json.dumps(task)+'\n' for task in tasks))

    def score(self,*args,**kwargs):
        selected=kwargs.get('metrics',list(self.values));self.calls.append(list(selected))
        return {**{key:self.values[key] if key in selected else None for key in self.values},
                'metric_status':{key:'ok' if key in selected else 'skipped' for key in self.values},'metric_errors':{}}

    def invoke(self,*extra,score=None,runtime=None):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),\
                patch.object(evaluate,'score_case',side_effect=score or self.score),\
                patch.object(evaluate,'_configure_runtime',side_effect=runtime or (lambda _:[])):
            return evaluate.main(['--dataset-root',str(self.data),'--predictions',str(self.predictions),
                                  '--output',str(self.output),*extra])

    def report(self):return json.loads(self.output.read_text())

    def test_complete_successes_skip_all_computation_and_runtime_initialization(self):
        self.assertEqual(self.invoke(),0);self.assertEqual(len(self.calls),2)
        self.assertEqual(self.invoke(score=lambda *a,**k:self.fail('Cached metrics recomputed'),
                                     runtime=lambda _:self.fail('Cached run initialized GPU/metrics')),0)
        self.assertEqual(self.report()['resume'],{'enabled':True,'reused_metrics':8,'computed_metrics':0,'skipped_cases':2})

    def test_partial_failures_only_compute_failed_metric(self):
        def partial(*args,**kwargs):
            result=self.score(*args,**kwargs)
            if len(self.calls)==1:
                result['lpips']=None;result['metric_status']['lpips']='error';result['metric_errors']['lpips']='fixture unavailable'
            return result
        self.assertEqual(self.invoke(score=partial),1);self.calls=[]
        self.assertEqual(self.invoke(),0);self.assertEqual(self.calls,[['lpips']])
        self.assertEqual(self.report()['resume']['reused_metrics'],7)
        self.assertEqual(self.report()['resume']['computed_metrics'],1)
        self.assertEqual(self.report()['results'][0]['ssim'],.9)

    def test_failed_retry_preserves_successful_axes_and_keeps_failed_mean_null(self):
        def partial(*args,**kwargs):
            result=self.score(*args,**kwargs)
            result['lpips']=None;result['metric_status']['lpips']='error';result['metric_errors']['lpips']='fixture failure'
            return result
        self.assertEqual(self.invoke(score=partial),1);self.calls=[]
        self.assertEqual(self.invoke(score=partial),1)
        self.assertEqual(self.calls,[['lpips'],['lpips']])
        self.assertIsNone(self.report()['summary']['metrics']['avg_lpips'])
        self.assertEqual(self.report()['summary']['metrics']['avg_ssim'],.9)

    def test_prediction_or_mask_change_invalidates_only_affected_case(self):
        self.assertEqual(self.invoke(),0);self.calls=[]
        Image.new('RGB',(64,64),'red').save(self.predictions/'b.png')
        self.assertEqual(self.invoke(),0);self.assertEqual(len(self.calls),1)
        self.calls=[]
        mask=np.zeros((64,64),np.uint8);mask[2:7,3:8]=255
        Image.fromarray(mask).save(self.data/'cases/a/target_region_mask.png')
        self.assertEqual(self.invoke(),0);self.assertEqual(len(self.calls),1)

    def test_missing_input_does_not_erase_historical_success(self):
        self.assertEqual(self.invoke(),0);original=(self.predictions/'a.png').read_bytes()
        (self.predictions/'a.png').unlink();self.calls=[]
        self.assertEqual(self.invoke(),1);self.assertEqual(self.calls,[])
        (self.predictions/'a.png').write_bytes(original)
        self.assertEqual(self.invoke(),0);self.assertEqual(self.calls,[])

    def test_subset_selection_preserves_other_cases_for_later_full_resume(self):
        self.assertEqual(self.invoke(),0);self.calls=[]
        self.assertEqual(self.invoke('--ids','a'),0);self.assertEqual(self.report()['expected'],1)
        self.assertEqual(self.invoke(),0);self.assertEqual(self.report()['expected'],2)
        self.assertEqual(self.calls,[])

    def test_verified_old_wrapper_reports_migrate_without_recomputing(self):
        self.assertEqual(self.invoke(),0)
        document=self.report()
        for name,hashes in evaluate.COMPATIBLE_WRAPPERS.items():
            document['protocol']['evaluator_sources'][name]['sha256']=next(iter(hashes))
        self.output.write_text(json.dumps(document));evaluate.cache_path(self.output).unlink();self.calls=[]
        self.assertEqual(self.invoke(),0);self.assertEqual(self.calls,[])

    def test_alpha_policy_diagnostics_change_keeps_validated_rgb_scores(self):
        self.assertEqual(self.invoke(),0)
        document=self.report()
        for row in document['results']:
            row['provenance']['alpha_policy']='Legacy near-opaque alpha gate'
        document['protocol']['evaluator_sources']['evaluate.py']['sha256']='3ec5a54f267b25c4363ab375790446356108805e8f52b61febfe5069d1fbc0f1'
        document['protocol']['evaluator_sources']['output_adapter.py']['sha256']='0d22becf61d07def54a37a4f5615b86cbc91de757ee571c359360021a3fe0169'
        self.output.write_text(json.dumps(document));evaluate.cache_path(self.output).unlink();self.calls=[]
        self.assertEqual(self.invoke(),0);self.assertEqual(self.calls,[])
        self.assertEqual(self.report()['resume']['reused_metrics'],8)

    def test_unknown_code_or_device_changes_do_not_reuse_scores(self):
        self.assertEqual(self.invoke(),0)
        document=self.report();document['protocol']['evaluator_sources']['evaluate.py']['sha256']='0'*64
        self.output.write_text(json.dumps(document));evaluate.cache_path(self.output).unlink();self.calls=[]
        self.assertEqual(self.invoke(),0);self.assertEqual(len(self.calls),2)
        self.calls=[]
        self.assertEqual(self.invoke('--device','cuda'),0);self.assertEqual(len(self.calls),2)

    def test_interruption_after_durable_cache_before_report_preserves_scores(self):
        original=evaluate._dump;interrupted=False
        def interrupt(path,value):
            nonlocal interrupted
            if path==self.output and not interrupted and any(row['status']=='ok' for row in value.get('results',[])):
                interrupted=True;raise KeyboardInterrupt()
            return original(path,value)
        with patch.object(evaluate,'_dump',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):self.invoke()
        self.calls=[];self.assertEqual(self.invoke(),0);self.assertEqual(len(self.calls),1)

    def test_force_recompute_and_corrupt_report_preservation(self):
        self.assertEqual(self.invoke(),0);self.calls=[]
        self.assertEqual(self.invoke('--no-resume'),0);self.assertEqual(len(self.calls),2)
        self.output.write_text('{broken');self.calls=[]
        self.assertEqual(self.invoke(),2);self.assertEqual(self.output.read_text(),'{broken');self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
