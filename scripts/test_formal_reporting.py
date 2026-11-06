"""Formal, multi-model reporting with real files and an offline judge stub."""
import concurrent.futures
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import compare_models
import evaluate
from evaluation_binding import canonical_hash
import formal_report
import mllm_evaluate as judge
from output_adapter import instruction_sha256,prepare_rgb_record,sha256_file


class FormalReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.dataset=self.root/'dataset';self.runs=self.root/'runs';self.runs.mkdir()
        self.tasks=[]
        pixels=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
        for cid in ('a','b'):
            folder=self.dataset/'cases'/cid;folder.mkdir(parents=True)
            Image.fromarray(pixels).save(folder/'source.png');Image.fromarray(pixels).save(folder/'target.png')
            mask=np.zeros((64,64),np.uint8);mask[5:15,6:20]=255
            for name in ('mask.png','target_region_mask.png'):Image.fromarray(mask).save(folder/name)
            self.tasks.append({'id':cid,'source_id':cid,'image':f'cases/{cid}/source.png','operation':'delete',
                               'domain':'poster','instruction':"Delete 'a'.",'output_filename':cid+'.png'})
        (self.dataset/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))
        (self.dataset/'summary.json').write_text('{"version":"fixture"}\n')
        self.summary_before=(self.dataset/'summary.json').read_bytes()
        self.calls=0

    def tearDown(self):
        self.assertEqual((self.dataset/'summary.json').read_bytes(),self.summary_before)
        self.temporary.cleanup()

    @staticmethod
    def pixel_scores(*args,**kwargs):
        values={'mse':10.,'psnr':38.1308,'ssim':.99,'lpips':.01}
        return {**values,'metric_status':dict.fromkeys(values,'ok'),'metric_errors':{}}

    def transport(self,endpoint,key,payload,timeout):
        self.calls+=1
        prompt=payload['messages'][0]['content']
        dimension=next(d for d in ('IF','TA','VC','LP','SE') if '{"'+d+'": X' in prompt)
        return {'choices':[{'message':{'content':json.dumps({dimension:4,'rationale':'Offline judged against image and prompt.'})},'finish_reason':'stop'}]}

    def create_run(self,model='model-a',ids=('a',),*,scored=True,judge_model='mock-judge',steps=4,fail_dimension=None):
        run=self.runs/model;run.mkdir()
        content={'version':'SYNTHETIC_TEST_CONTENT_NOT_MODEL'};content['sha256']=canonical_hash(content)
        protocol={'case_ids':list(ids),'model_id':model,'run_id':'fixture-run','steps':steps,
            'generation_protocol_version':'multi-image-edit-generation-v1','content_provenance':content,
            'resolution_protocol':'native-pad32-v1'}
        protocol['protocol_sha256']=canonical_hash(protocol)
        (run/'protocol.json').write_text(json.dumps(protocol))
        records=[]
        for task in self.tasks:
            if task['id'] not in ids:continue
            folder=run/task['id'];folder.mkdir();raw=folder/'raw.png';rgb=folder/'prediction.png'
            with Image.open(self.dataset/task['image']) as source:Image.fromarray(np.dstack([np.asarray(source),np.full((64,64),255,np.uint8)])).save(raw)
            metadata={'id':task['id'],'model_id':model,'run_id':'fixture-run','source_size':[64,64],
                'canvas_size':[64,64],'evaluation_roi':[0,0,64,64],
                'padding':{'left':0,'top':0,'right':0,'bottom':0,'mode':'none'},
                'source_sha256':sha256_file(self.dataset/task['image']),'instruction_sha256':instruction_sha256(task),
                'protocol_sha256':protocol['protocol_sha256'],'content_provenance_sha256':content['sha256']}
            records.append(prepare_rgb_record(raw,rgb,metadata))
        manifest=run/'generation_manifest.json'
        manifest.write_text(json.dumps({'schema_version':1,'status':'complete','expected_ids':list(ids),
            'model_id':model,'run_id':'fixture-run','generation_protocol_version':protocol['generation_protocol_version'],
            'content_provenance_sha256':content['sha256'],'protocol_sha256':protocol['protocol_sha256'],'records':records}))
        with contextlib.redirect_stdout(io.StringIO()),patch.object(evaluate,'score_case',side_effect=self.pixel_scores),patch.object(evaluate,'_configure_runtime',return_value=[]):
            self.assertEqual(evaluate.main(['--dataset-root',str(self.dataset),'--generation-manifest',str(manifest),
                '--ids',','.join(ids),'--output',str(run/'pixel_scores.json')]),0)
        config=run/'judge_config.json';config.write_text(json.dumps({'model':judge_model,'api_key_env':'OFFLINE_JUDGE_KEY',
            'base_url':'https://judge.invalid/v1','max_retries':0}))
        args=['--dataset',str(self.dataset),'--generation-manifest',str(manifest),'--ids',','.join(ids),
              '--config',str(config),'--output',str(run/('mllm_scores.json' if scored else 'mllm_pending.json'))]
        if not scored:args.append('--dry-run')
        def transport(*values):
            response=self.transport(*values)
            content=response['choices'][0]['message']['content']
            if fail_dimension and fail_dimension in json.loads(content):
                response['choices'][0]['message']['content']='invalid judge JSON'
            return response
        with patch.dict(os.environ,{'OFFLINE_JUDGE_KEY':'OFFLINE_NOT_REAL'},clear=True):
            report,code=judge.evaluate(judge.parser().parse_args(args),transport=transport,sleeper=lambda _:None)
        self.assertEqual(code,1 if scored and fail_dimension else 0)
        return run

    def test_real_formal_scores_are_accepted_without_an_api_call(self):
        run=self.create_run();count=self.calls
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,0);self.assertTrue(report['full_evaluation_complete'])
        self.assertEqual(report['metrics']['IF'],4);self.assertEqual(self.calls,count)
        self.assertEqual(report['model_id'],'model-a');self.assertEqual(report['expected_cases'],1)

    def test_text_joint_rate_uses_all_selected_cases_and_is_exported_to_csv(self):
        run=self.create_run(ids=('a','b'));path=run/'mllm_scores.json';saved=json.loads(path.read_text())
        next(row for row in saved['results'] if row['id']=='b')['dimensions']['TA']['score']=3
        saved['summary']=judge.summarize(saved['results']);path.write_text(json.dumps(saved))
        before=path.read_bytes();calls=self.calls
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,0);self.assertEqual(self.calls,calls);self.assertEqual(path.read_bytes(),before)
        joint=report['joint_qualification']
        self.assertEqual(joint['dimensions'],['IF','TA','VC','LP']);self.assertEqual(joint['threshold_each'],4)
        self.assertEqual(joint['expected'],2);self.assertEqual(joint['qualified'],1)
        self.assertTrue(joint['complete']);self.assertEqual(joint['rate'],.5)
        self.assertEqual(report['metrics']['joint_qualification_rate'],.5)
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,0);self.assertEqual(result['results'][0]['joint_qualification'],joint)
        self.assertEqual(result['results'][0]['metrics']['joint_qualification_rate'],.5)
        output=self.root/'text_comparison.csv';compare_models.write_csv(output,result)
        with output.open() as stream:row=next(csv.DictReader(stream))
        self.assertEqual(row['joint_qualification_rate'],'0.5')
        self.assertEqual(row['joint_qualified'],'1');self.assertEqual(row['joint_expected'],'2')

    def test_incomplete_text_joint_keeps_full_denominator_and_null_rate(self):
        run=self.create_run(ids=('a','b'));path=run/'mllm_scores.json';saved=json.loads(path.read_text())
        failed=next(row for row in saved['results'] if row['id']=='b')
        failed['dimensions']['TA'].update(status='error',score=None,error='Offline judge unavailable')
        failed['status']='partial_error';saved['status']='incomplete'
        saved['summary']=judge.summarize(saved['results']);path.write_text(json.dumps(saved))
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertTrue(report['semantic_inputs_valid'])
        joint=report['joint_qualification']
        self.assertEqual(joint['expected'],2);self.assertEqual(joint['scored'],1)
        self.assertEqual(joint['qualified'],1);self.assertFalse(joint['complete']);self.assertIsNone(joint['rate'])
        self.assertIsNone(report['metrics']['joint_qualification_rate'])
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,1);self.assertEqual(result['results'][0]['joint_qualification']['expected'],2)
        output=self.root/'incomplete_text.csv';compare_models.write_csv(output,result)
        with output.open() as stream:row=next(csv.DictReader(stream))
        self.assertEqual(row['joint_qualification_rate'],'');self.assertEqual(row['joint_expected'],'2')
        path.unlink();missing,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertEqual(missing['joint_qualification']['expected'],2)
        self.assertIsNone(missing['joint_qualification']['qualified']);self.assertIsNone(missing['joint_qualification']['rate'])

    def test_optional_se_score_or_failure_does_not_enter_four_dimension_joint(self):
        self.tasks[0]['knowledge_prompt']='Verify the requested factual content.'
        (self.dataset/'tasks.jsonl').write_text(''.join(json.dumps(task)+'\n' for task in self.tasks))
        run=self.create_run(ids=('a','b'));path=run/'mllm_scores.json';saved=json.loads(path.read_text())
        row=next(row for row in saved['results'] if row['id']=='a')
        row['dimensions']['SE']['score']=1
        saved['summary']=judge.summarize(saved['results']);path.write_text(json.dumps(saved))
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,0);self.assertEqual(report['metrics']['SE'],1)
        self.assertEqual(report['joint_qualification']['expected'],2)
        self.assertEqual(report['joint_qualification']['qualified'],2)
        self.assertEqual(report['metrics']['joint_qualification_rate'],1)
        row['dimensions']['SE'].update(status='error',score=None,error='Optional SE unavailable')
        row['status']='partial_error';saved['status']='incomplete'
        saved['summary']=judge.summarize(saved['results']);path.write_text(json.dumps(saved))
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['semantic_complete'])
        self.assertTrue(report['joint_qualification']['complete'])
        self.assertEqual(report['metrics']['joint_qualification_rate'],1)

    def test_dryrun_and_pixel_only_cannot_claim_full_semantic_completion(self):
        run=self.create_run(scored=False)
        dry,code=formal_report.evaluate_run(run,self.dataset,mode='dry-run')
        self.assertEqual(code,0);self.assertTrue(dry['requested_mode_valid']);self.assertFalse(dry['complete'])
        self.assertFalse(dry['semantic_complete']);self.assertIsNone(dry['metrics']['IF'])
        self.assertIsNone(dry['metrics']['joint_qualification_rate']);self.assertFalse(dry['joint_qualification']['complete'])
        pixel,code=formal_report.evaluate_run(run,self.dataset,mode='pixel-only')
        self.assertEqual(code,0);self.assertTrue(pixel['complete']);self.assertFalse(pixel['full_evaluation_complete'])
        self.assertIsNone(pixel['metrics']['joint_qualification_rate'])
        self.assertEqual(self.calls,0)

    def test_pending_judge_cannot_pass_formal_completion(self):
        run=self.create_run(scored=False)
        (run/'mllm_scores.json').write_bytes((run/'mllm_pending.json').read_bytes())
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['complete']);self.assertIsNone(report['metrics']['IF'])
        self.assertEqual(report['semantic_coverage']['IF']['expected'],1)

    def test_all_requires_the_entire_dataset_not_only_selected_case(self):
        run=self.create_run()
        with self.assertRaisesRegex(ValueError,'case set differs'):
            formal_report.evaluate_run(run,self.dataset,expected_ids='all')

    def test_tampered_judge_summary_is_rejected(self):
        run=self.create_run();path=run/'mllm_scores.json';saved=json.loads(path.read_text())
        saved['summary']['criteria']['IF']['mean']=5;path.write_text(json.dumps(saved))
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['semantic_complete'])
        self.assertTrue(any('summary differs' in error for error in report['validation_errors']))
        self.assertIsNone(report['metrics']['joint_qualification_rate'])
        self.assertIsNone(report['joint_qualification']['qualified'])

    def test_changed_mask_invalidates_formal_pixel_track(self):
        run=self.create_run();Image.new('L',(64,64),0).save(self.dataset/'cases/a/mask.png')
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['pixel_complete']);self.assertIsNone(report['metrics']['mse'])
        self.assertEqual(report['metrics']['joint_qualification_rate'],1)

    def test_model_specific_sampling_parameters_do_not_prevent_comparison(self):
        self.create_run('model-a',steps=4);self.create_run('model-b',steps=40)
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,0);self.assertTrue(result['complete']);self.assertEqual(result['model_runs'],2)
        self.assertEqual({row['model_id'] for row in result['results']},{'model-a','model-b'})
        output=self.root/'comparison.csv';compare_models.write_csv(output,result)
        self.assertIn('IF_scored',output.read_text());self.assertIn('model-a',output.read_text())

    def test_different_judges_are_not_merged(self):
        self.create_run('model-a');self.create_run('model-b',judge_model='other-judge')
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,2);self.assertFalse(result['complete'])
        self.assertIn('Models differ in judge protocol',result['validation_errors'])
        self.assertFalse(result['comparable']);self.assertIsNone(result['comparison_protocol'])
        for row in result['results']:
            self.assertEqual(row['metrics']['mse'],10);self.assertEqual(row['metrics']['IF'],4)
            self.assertEqual(row['metrics']['joint_qualification_rate'],1)
            self.assertTrue(row['complete']);self.assertFalse(row['comparable'])
            self.assertIsNotNone(row['comparison_protocol']['judge'])
        output=self.root/'incompatible.csv';compare_models.write_csv(output,result)
        with output.open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]['IF'],'4.0');self.assertEqual(rows[0]['comparable'],'False')
        self.assertIn('judge protocol',rows[0]['comparison_errors'])

    def test_different_case_subsets_are_not_merged(self):
        self.create_run('model-a',ids=('a',));self.create_run('model-b',ids=('b',))
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,2);self.assertIn('Models differ in case_ids',result['validation_errors'])
        self.assertIsNone(result['expected_cases']);self.assertIsNone(result['scope'])
        self.assertTrue(all(row['metrics']['IF']==4 for row in result['results']))

    def test_corrupt_model_protocol_does_not_discard_valid_model_in_cli_exports(self):
        self.create_run('model-a');bad=self.create_run('model-b')
        (bad/'protocol.json').write_text('{broken')
        calls=self.calls;output=self.root/'comparison'
        with contextlib.redirect_stdout(io.StringIO()):
            code=compare_models.main(['--runs-root',str(self.runs),'--dataset-root',str(self.dataset),
                                      '--output-dir',str(output)])
        self.assertEqual(code,2);self.assertEqual(self.calls,calls)
        result=json.loads((output/'comparison.json').read_text())
        self.assertFalse(result['complete']);self.assertEqual(result['model_runs'],2)
        good=next(row for row in result['results'] if row['model_id']=='model-a')
        failed=next(row for row in result['results'] if row['model_id']=='model-b')
        self.assertTrue(good['complete']);self.assertEqual(good['metrics']['IF'],4)
        self.assertEqual(failed['status'],'invalid_run');self.assertFalse(failed['complete'])
        self.assertTrue(failed['validation_errors']);self.assertIsNone(failed['metrics']['mse'])
        with (output/'comparison.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual({row['model_id'] for row in rows},{'model-a','model-b'})
        self.assertTrue(next(row for row in rows if row['model_id']=='model-b')['validation_errors'])

    def test_evaluation_exception_is_isolated_to_its_model(self):
        self.create_run('model-a',ids=('a',));self.create_run('model-b',ids=('b',))
        result,code=compare_models.compare_runs(self.runs,self.dataset,expected_ids='a')
        self.assertEqual(code,2);self.assertEqual(result['model_runs'],2)
        good=next(row for row in result['results'] if row['model_id']=='model-a')
        failed=next(row for row in result['results'] if row['model_id']=='model-b')
        self.assertEqual(good['metrics']['ssim'],.99);self.assertTrue(good['complete'])
        self.assertEqual(failed['status'],'invalid_run')
        self.assertIn('case set differs',failed['validation_errors'][0])

    def test_selected_missing_run_does_not_block_first_export_of_valid_model(self):
        self.create_run('model-a');(self.runs/'model-b').mkdir();output=self.root/'comparison'
        with contextlib.redirect_stdout(io.StringIO()):
            code=compare_models.main(['--runs-root',str(self.runs),'--dataset-root',str(self.dataset),
                                      '--models','model-a,model-b','--output-dir',str(output)])
        self.assertEqual(code,2)
        result=json.loads((output/'comparison.json').read_text())
        self.assertFalse(result['complete']);self.assertEqual(result['model_runs'],2)
        good=next(row for row in result['results'] if row['model_id']=='model-a')
        missing=next(row for row in result['results'] if row['model_id']=='model-b')
        self.assertTrue(good['complete']);self.assertEqual(good['metrics']['IF'],4)
        self.assertEqual(good['metrics']['joint_qualification_rate'],1)
        self.assertEqual(missing['status'],'missing_run');self.assertFalse(missing['complete'])
        self.assertIsNone(missing['run_dir']);self.assertTrue(missing['validation_errors'])
        self.assertTrue(all(value is None for value in missing['metrics'].values()))
        with (output/'comparison.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual({row['model_id'] for row in rows},{'model-a','model-b'})
        self.assertEqual(next(row for row in rows if row['model_id']=='model-b')['status'],'missing_run')

    def test_invalid_comparison_request_preserves_previous_exports(self):
        self.create_run();output=self.root/'comparison'
        args=['--runs-root',str(self.runs),'--dataset-root',str(self.dataset),'--output-dir',str(output)]
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(compare_models.main(args),0)
        previous={name:(output/name).read_bytes() for name in ('comparison.json','comparison.csv')}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_models.main([*args,'--models','missing-model']),2)
        self.assertEqual({name:(output/name).read_bytes() for name in previous},previous)
        error=json.loads((output/'comparison.error.json').read_text())
        self.assertEqual(error['status'],'invalid_request')

    def test_missing_model_pixel_scores_stay_incomplete_without_fake_zero(self):
        self.create_run('model-a');bad=self.create_run('model-b');(bad/'pixel_scores.json').unlink()
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,1);self.assertEqual(result['status'],'incomplete')
        row=next(row for row in result['results'] if row['model_id']=='model-b')
        self.assertIsNone(row['metrics']['mse']);self.assertFalse(row['complete'])
        self.assertEqual(next(row for row in result['results'] if row['model_id']=='model-a')['metrics']['IF'],4)

    def test_alignment_failure_keeps_complete_ssim_lpips_in_formal_and_comparison(self):
        run=self.create_run()
        partial={'mse':None,'psnr':None,'ssim':.99,'lpips':.01,
            'metric_status':{'mse':'error','psnr':'error','ssim':'ok','lpips':'ok'},
            'metric_errors':{'mse':'Official SIFT alignment failed','psnr':'Official SIFT alignment failed'}}
        with contextlib.redirect_stdout(io.StringIO()),patch.object(evaluate,'score_case',return_value=partial),patch.object(evaluate,'_configure_runtime',return_value=[]):
            code=evaluate.main(['--dataset-root',str(self.dataset),'--generation-manifest',str(run/'generation_manifest.json'),
                '--ids','a','--output',str(run/'pixel_scores.json'),'--no-resume'])
        self.assertEqual(code,1)
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertTrue(report['pixel_inputs_valid']);self.assertFalse(report['complete'])
        self.assertIsNone(report['metrics']['mse']);self.assertIsNone(report['metrics']['psnr'])
        self.assertEqual(report['metrics']['ssim'],.99);self.assertEqual(report['metrics']['lpips'],.01)
        self.assertEqual(report['pixel_coverage']['mse']['failed'],1)
        self.assertTrue(report['pixel_coverage']['ssim']['complete'])
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,1);self.assertFalse(result['complete'])
        self.assertEqual(result['results'][0]['metrics']['ssim'],.99)
        self.assertIsNone(result['results'][0]['metrics']['psnr'])
        # Current input binding is mandatory even for otherwise complete axes.
        Image.new('L',(64,64),0).save(self.dataset/'cases/a/mask.png')
        stale,_=formal_report.evaluate_run(run,self.dataset)
        self.assertFalse(stale['pixel_inputs_valid'])
        self.assertTrue(all(stale['metrics'][name] is None for name in formal_report.PIXEL_NAMES))

    def test_one_api_dimension_failure_keeps_other_complete_means(self):
        run=self.create_run(fail_dimension='TA')
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertTrue(report['semantic_inputs_valid']);self.assertFalse(report['semantic_complete'])
        self.assertIsNone(report['metrics']['TA'])
        for name in ('IF','VC','LP'):self.assertEqual(report['metrics'][name],4)
        self.assertEqual(report['semantic_coverage']['TA']['coverage'],0)
        self.assertEqual(report['semantic_coverage']['IF']['coverage'],1)
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,1);self.assertFalse(result['complete'])
        self.assertIsNone(result['results'][0]['metrics']['TA'])
        self.assertEqual(result['results'][0]['metrics']['IF'],4)

    def test_parallel_per_run_reports_do_not_touch_global_summary(self):
        runs=[self.create_run('model-a'),self.create_run('model-b')]
        def run(path):return formal_report.main(['--run-dir',str(path),'--dataset-root',str(self.dataset)])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(run,runs)),[0,0])
        self.assertTrue(all((run/'report.json').exists() for run in runs))
        self.assertEqual((self.dataset/'summary.json').read_bytes(),self.summary_before)


class AssetFormalReportingTests(unittest.TestCase):
    """Asset semantics must remain distinct through the final public table."""
    tearDown=FormalReportingTests.tearDown
    pixel_scores=staticmethod(FormalReportingTests.pixel_scores)
    create_run=FormalReportingTests.create_run

    def setUp(self):
        FormalReportingTests.setUp(self)
        refs=[]
        for task in self.tasks:
            task.update(input_protocol='asset-edit-single-source-v1',asset_family_id='family-'+task['id'],
                        instruction='Delete the photo near the center.')
            folder=self.dataset/'cases'/task['id']
            with Image.open(folder/'mask.png') as mask:
                mask.save(folder/'allowed_edit_mask.png')
                mask.save(folder/'eval_mask.png')
            refs.append({'id':task['id'],'pixel_preservation_mask':f'cases/{task["id"]}/allowed_edit_mask.png'})
        (self.dataset/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))
        (self.dataset/'references.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))

    def transport(self,endpoint,key,payload,timeout):
        self.calls+=1
        prompt=payload['messages'][0]['content']
        dimension=next(d for d in ('IF','OA','VC','LP') if '{"'+d+'": X' in prompt)
        return {'choices':[{'message':{'content':json.dumps({dimension:4,'rationale':'Offline asset evidence.',
             'evidence':{'source':'Source asset.','candidate':'Edited asset.','change':'Requested edit.',
                         'target_match':'yes','operation':'done','defects':[]}})},
                            'finish_reason':'stop'}]}

    def test_asset_profile_reports_oa_and_joint_rate_without_text_ta_or_se(self):
        run=self.create_run()
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,0);self.assertTrue(report['full_evaluation_complete'])
        self.assertEqual(report['track'],'asset');self.assertEqual(report['metrics']['OA'],4)
        self.assertNotIn('TA',report['metrics']);self.assertNotIn('SE',report['metrics'])
        self.assertEqual(report['joint_qualification']['rate'],1)
        self.assertEqual(report['metrics']['joint_qualification_rate'],1)
        self.assertEqual(report['by_operation']['delete']['expected'],1)
        result,code=compare_models.compare_runs(self.runs,self.dataset)
        self.assertEqual(code,0);self.assertEqual(result['track'],'asset')
        output=self.root/'asset_comparison.csv';compare_models.write_csv(output,result)
        headers=output.read_text().splitlines()[0].split(',')
        self.assertIn('OA',headers);self.assertNotIn('TA',headers)
        self.assertIn('joint_qualification_rate',headers);self.assertIn('joint_expected',headers)
        with output.open() as stream:row=next(csv.DictReader(stream))
        self.assertEqual(row['joint_qualification_rate'],'1.0')
        self.assertEqual(row['joint_qualified'],'1');self.assertEqual(row['joint_expected'],'1')

    def test_text_dimension_cannot_be_silently_substituted_into_asset_report(self):
        run=self.create_run();path=run/'mllm_scores.json';saved=json.loads(path.read_text())
        row=saved['results'][0];row['dimensions']['TA']=row['dimensions'].pop('OA')
        path.write_text(json.dumps(saved))
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['semantic_inputs_valid'])
        self.assertTrue(any('dimensions differ' in e for e in report['validation_errors']))

    def test_asset_api_failure_blocks_joint_rate_and_all_conditional_means(self):
        run=self.create_run(fail_dimension='OA')
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertTrue(report['semantic_inputs_valid'])
        self.assertFalse(report['semantic_complete']);self.assertIsNone(report['joint_qualification']['rate'])
        self.assertTrue(all(report['metrics'][d] is None for d in ('IF','OA','VC','LP')))

    def test_allowed_region_change_invalidates_only_asset_pixels_not_full_image_judge(self):
        run=self.create_run();Image.new('L',(64,64),255).save(self.dataset/'cases/a/allowed_edit_mask.png')
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['pixel_inputs_valid'])
        self.assertTrue(report['semantic_inputs_valid'])

    def test_gt_union_change_does_not_replace_source_defined_background_mask(self):
        run=self.create_run();Image.new('L',(64,64),255).save(self.dataset/'cases/a/eval_mask.png')
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,0);self.assertTrue(report['pixel_inputs_valid'])

    def test_missing_asset_output_counts_in_joint_denominator_without_fake_full_completion(self):
        run=self.create_run(ids=('a','b'))
        manifest=json.loads((run/'generation_manifest.json').read_text())
        missing=next(r for r in manifest['records'] if r['id']=='b')
        Path(missing['raw_output']).unlink()
        args=judge.parser().parse_args(['--dataset',str(self.dataset),'--generation-manifest',str(run/'generation_manifest.json'),
            '--ids','a,b','--config',str(run/'judge_config.json'),'--output',str(run/'mllm_scores.json'),'--no-resume'])
        with patch.dict(os.environ,{'OFFLINE_JUDGE_KEY':'OFFLINE_NOT_REAL'},clear=True):
            scored,code=judge.evaluate(args,transport=self.transport,sleeper=lambda _:None)
        self.assertEqual(code,0);self.assertEqual(scored['summary']['model_failures'],1)
        report,code=formal_report.evaluate_run(run,self.dataset)
        self.assertEqual(code,1);self.assertFalse(report['generation_complete'])
        self.assertFalse(report['full_evaluation_complete']);self.assertTrue(report['semantic_complete'])
        self.assertEqual(report['joint_qualification']['expected'],2)
        self.assertEqual(report['joint_qualification']['rate'],.5)
        self.assertEqual(report['metrics']['OA'],4)
        self.assertEqual(report['semantic_summary']['valid_output_coverage'],.5)


if __name__=='__main__':unittest.main()
