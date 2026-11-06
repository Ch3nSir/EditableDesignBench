"""Asset judging contracts, fixed views, failure denominators and resume; offline."""
import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import asset_judge_profile as asset
import mllm_evaluate as judge


class AssetJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.predictions=self.root/'secret_edit_model_23';self.predictions.mkdir()
        self.tasks=[];references=[]
        for index,operation in enumerate(asset.METHODS):
            cid=f'asset_{index}_{operation}';folder=self.root/'cases'/cid;folder.mkdir(parents=True)
            Image.new('RGB',(640,480),'white').save(folder/'source.png')
            Image.new('RGB',(640,480),'blue').save(self.predictions/(cid+'.png'))
            mask=Image.new('L',(640,480),0)
            ImageDraw.Draw(mask).rectangle((90,90,109,109),fill=255);mask.save(folder/'allowed_edit_mask.png')
            self.tasks.append({'id':cid,'source_id':str(index),'asset_family_id':'family_'+str(index),
                'operation':operation,'instruction':f'{operation.title()} the green vase.',
                'image':f'cases/{cid}/source.png','output_filename':cid+'.png',
                'input_protocol':asset.INPUT_PROTOCOL})
            references.append({'id':cid,'pixel_preservation_mask':f'cases/{cid}/allowed_edit_mask.png',
                               'target':'a_reference_target_that_does_not_exist.png'})
        self.write_tasks()
        (self.root/'references.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in references))
        self.environment=patch.dict(os.environ,{'OPENAI_API_KEY':'OFFLINE_FIXTURE',
            'OPENAI_API_URL':'https://judge.invalid/v1','OPENAI_MODEL':'offline-judge'},clear=True)
        self.environment.start();self.addCleanup(self.environment.stop)

    def write_tasks(self):
        (self.root/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))

    def args(self,*extra):
        return judge.parser().parse_args(['--dataset',str(self.root),'--predictions',str(self.predictions),
            '--output',str(self.root/'scores.json'),'--workers','2','--max-retries','0','--no-progress',*extra])

    @staticmethod
    def dimension(payload):
        return next(d for d in asset.DIMENSIONS if '{"'+d+'": X' in payload['messages'][0]['content'])

    @classmethod
    def success(cls,endpoint,key,payload,timeout):
        return {'choices':[{'message':{'content':json.dumps({cls.dimension(payload):4,'rationale':'Visible evidence checked.',
                'evidence':{'source':'Original green vase in the requested slot.',
                            'candidate':'Requested result in the same slot.',
                            'change':'The requested unit was edited.',
                            'target_match':'yes','operation':'done','defects':[]}})},
                            'finish_reason':'stop'}]}

    @staticmethod
    def decoded_views(payload):
        urls=[x['image_url']['url'] for x in payload['messages'][1]['content'] if x['type']=='image_url']
        return [Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))) for url in urls]

    def test_evidence_is_required_and_scores_are_not_silently_clamped(self):
        args=self.args('--ids',self.tasks[0]['id'])
        def unsupported(e,k,p,t):
            return {'choices':[{'message':{'content':json.dumps({self.dimension(p):5,'rationale':'Looks good.'})}}]}
        report,code=judge.evaluate(args,transport=unsupported)
        self.assertEqual(code,1)
        self.assertEqual(report['summary']['errors'],1)
        self.assertTrue(all(v['score'] is None for v in report['results'][0]['dimensions'].values()))
        self.assertEqual(report['results'][0]['dimensions']['IF']['error'],'Asset judge response requires evidence')

    def test_self_contradictory_response_is_accepted_once_with_original_score(self):
        args=self.args('--ids',self.tasks[0]['id'],'--max-retries','1')
        attempts={d:0 for d in asset.DIMENSIONS}
        def transport(e,k,p,t):
            d=self.dimension(p);attempts[d]+=1
            result=self.success(e,k,p,t)
            data=json.loads(result['choices'][0]['message']['content'])
            if d=='IF' and attempts[d]==1:
                data['IF']=5;data['evidence']['target_match']='no'
            result['choices'][0]['message']['content']=json.dumps(data)
            return result
        report,code=judge.evaluate(args,transport=transport,sleeper=lambda _:None)
        self.assertEqual(code,0);self.assertEqual(report['api_calls_made'],4)
        self.assertEqual(report['results'][0]['dimensions']['IF']['score'],5)
        self.assertEqual(report['results'][0]['dimensions']['IF']['attempts'],1)
        self.assertIn('IF_score_target_disagreement',report['results'][0]['consistency_flags'])
        self.assertEqual(report['summary']['consistency_review']['cases'],1)
        self.assertFalse(report['response_validation']['retry_on_semantic_disagreement'])

    def test_every_previous_semantic_rejection_is_review_only(self):
        evidence={'source':'Original vase.','candidate':'Same vase.','change':'No movement.',
                  'target_match':'yes','operation':'done','defects':[]}
        for dimension,score,changes,flag in [
            ('IF',5,{'target_match':'no'},'IF_score_target_disagreement'),
            ('OA',3,{'target_match':'no'},'OA_score_target_disagreement'),
            ('IF',4,{'target_match':'uncertain'},'IF_high_score_uncertain_target'),
            ('OA',5,{'target_match':'uncertain'},'OA_high_score_uncertain_target'),
            ('IF',2,{'operation':'not_done'},'IF_score_operation_disagreement'),
            ('IF',5,{'operation':'wrong'},'IF_score_operation_disagreement'),
            ('IF',4,{'operation':'partial'},'IF_high_score_incomplete_operation'),
            ('IF',4,{'operation':'uncertain'},'IF_high_score_incomplete_operation')]:
            with self.subTest(dimension=dimension,changes=changes):
                payload={dimension:score,'rationale':'Visible result.','evidence':dict(evidence,**changes)}
                parsed=judge.parse_score(json.dumps(payload),dimension,asset_response=True)
                self.assertEqual(parsed[dimension],score)
                flags=asset.consistency_flags({dimension:{'status':'scored','score':score,'evidence':parsed['evidence']}})
                self.assertIn(flag,flags)

    def test_old_successes_are_reused_when_retrying_a_previously_rejected_dimension(self):
        args=self.args('--ids',self.tasks[0]['id'])
        def response(e,k,p,t):
            result=self.success(e,k,p,t);d=self.dimension(p)
            if d=='IF':
                data=json.loads(result['choices'][0]['message']['content'])
                data['IF']=5;data['evidence']['target_match']='no'
                result['choices'][0]['message']['content']=json.dumps(data)
            return result
        real_validate=judge.asset_validation.validate_evidence
        def legacy_validate(data,dimension):
            evidence=real_validate(data,dimension)
            if dimension=='IF' and evidence['target_match']=='no' and data[dimension]>1:
                raise ValueError('Legacy semantic rejection')
            return evidence
        with patch.object(judge.asset_validation,'validate_evidence',side_effect=legacy_validate):
            before,code=judge.evaluate(args,transport=response)
        self.assertEqual(code,1)
        after,code=judge.evaluate(args,transport=response)
        self.assertEqual(code,0);self.assertEqual(after['api_calls_made'],1)
        self.assertEqual(after['resume']['cached_dimensions'],3)
        self.assertEqual(before['results'][0]['fingerprint'],after['results'][0]['fingerprint'])
        for dimension in ('OA','VC','LP'):
            self.assertEqual(before['results'][0]['dimensions'][dimension],after['results'][0]['dimensions'][dimension])
        self.assertEqual(after['results'][0]['dimensions']['IF']['score'],5)

    def test_independent_axes_and_conflicting_observations_are_audited_without_score_changes(self):
        payload={'OA':5,'rationale':'Identity is intact despite no movement.',
                 'evidence':{'source':'Original vase.','candidate':'Same vase.', 'change':'No movement.',
                             'target_match':'yes','operation':'not_done','defects':[]}}
        self.assertEqual(judge.parse_score(json.dumps(payload),'OA',asset_response=True)['OA'],5)
        payload['IF']=5
        self.assertEqual(judge.parse_score(json.dumps(payload),'IF',asset_response=True)['IF'],5)
        del payload['IF']
        dimensions={'OA':{'status':'scored','score':5,'evidence':payload['evidence']},
                    'IF':{'status':'scored','score':0,'evidence':dict(payload['evidence'],target_match='no',operation='wrong')},
                    'VC':{'status':'scored','score':5,'evidence':dict(payload['evidence'],operation='done')}}
        self.assertEqual(asset.consistency_flags(dimensions),['conflicting_target_match','conflicting_operation_status'])
        self.assertEqual(dimensions['OA']['score'],5)
        unchanged={k:dict(v,evidence=dict(v['evidence'],operation='not_done')) for k,v in dimensions.items()}
        self.assertEqual(asset.consistency_flags(unchanged),[])
        payload['evidence']['operation']='done'
        self.assertEqual(judge.parse_score(json.dumps(payload),'OA',asset_response=True)['OA'],5)

    def test_auto_profile_four_independent_requests_exactly_two_native_full_images(self):
        payloads=[]
        def transport(e,k,p,t):payloads.append(p);return self.success(e,k,p,t)
        report,code=judge.evaluate(self.args(),transport=transport)
        self.assertEqual(code,0);self.assertEqual(len(payloads),20)
        self.assertEqual(judge.infer_profile(self.tasks),'asset')
        self.assertEqual(report['protocol']['criteria'],list(asset.DIMENSIONS))
        self.assertEqual(report['protocol']['message_template_revision'],asset.MESSAGE_TEMPLATE_REVISION)
        self.assertTrue(judge.validate_report_protocol(report['protocol'],self.tasks))
        self.assertFalse(judge.validate_report_protocol(None,self.tasks))
        self.assertFalse(judge.validate_report_protocol(report['protocol'],[None]))
        for payload in payloads:
            self.assertEqual(payload['temperature'],0);self.assertEqual(payload['max_tokens'],500)
            self.assertEqual(len(payload['messages']),2)
            views=self.decoded_views(payload)
            self.assertEqual([im.size for im in views],[(640,480),(640,480)])
            self.assertTrue(np.array_equal(np.asarray(views[0]),np.full((480,640,3),255,np.uint8)))
            text=json.dumps(payload['messages'])
            self.assertNotIn('secret_edit_model_23',text)
            self.assertNotIn('a_reference_target_that_does_not_exist',text)
            self.assertIn('MUST be a JSON number',payload['messages'][1]['content'][-1]['text'])
            self.assertNotIn('"<score 0-5>"',payload['messages'][1]['content'][-1]['text'])
        for row in report['results']:
            self.assertEqual(tuple(row['dimensions']),asset.DIMENSIONS)
            self.assertEqual(row['judge_views']['image_count'],2)
            self.assertFalse(row['judge_views']['uses_mask'])
            self.assertNotIn('crop_box',row['judge_views'])
        self.assertEqual(report['summary']['four_dimension_mean'],4)
        self.assertEqual(report['summary']['joint_qualification']['rate'],1)
        self.assertEqual(set(report['summary']['by_operation']),set(asset.METHODS))
        self.assertTrue(all(row['expected']==1 for row in report['summary']['by_operation'].values()))

    def test_cache_ignores_reference_and_masks_but_invalidates_changed_view_protocol_and_judge(self):
        args=self.args('--ids',self.tasks[0]['id'])
        first,_=judge.evaluate(args,transport=self.success)
        refs=self.root/'references.jsonl';rows=[json.loads(line) for line in refs.read_text().splitlines()]
        for row in rows:row['target']='another_nonexistent_GT.png'
        refs.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        cached,_=judge.evaluate(args,transport=lambda *a:self.fail('GT path changed paid no-reference scores'))
        self.assertEqual(cached['api_calls_made'],0)
        mask=self.root/rows[0]['pixel_preservation_mask']
        with Image.open(mask) as im:changed=im.copy()
        changed.putpixel((100,100),0);changed.save(mask)
        changed_mask,_=judge.evaluate(args,transport=lambda *a:self.fail('Unused mask invalidated full-image scores'))
        self.assertEqual(changed_mask['api_calls_made'],0)
        self.assertEqual(first['results'][0]['fingerprint'],changed_mask['results'][0]['fingerprint'])
        with patch.dict(asset.VIEW_PARAMETERS,{'full_images':'alternate-protocol-fixture'}):
            changed_crop,_=judge.evaluate(args,transport=self.success)
            self.assertEqual(changed_crop['api_calls_made'],4)
            self.assertNotEqual(changed_mask['results'][0]['fingerprint'],changed_crop['results'][0]['fingerprint'])
        args.model='another-judge'
        changed_judge,_=judge.evaluate(args,transport=self.success)
        self.assertEqual(changed_judge['api_calls_made'],4)
        self.assertNotEqual(changed_mask['results'][0]['fingerprint'],changed_judge['results'][0]['fingerprint'])

    def test_model_failure_is_joint_failure_api_failure_is_null_and_resume_only_retries_failed_dimension(self):
        missing=self.tasks[-1];(self.predictions/missing['output_filename']).unlink()
        def failing(e,k,p,t):
            if self.dimension(p)=='OA' and 'Delete the green vase.' in json.dumps(p['messages']):
                return {'choices':[{'message':{'content':'invalid JSON'},'finish_reason':'stop'}]}
            return self.success(e,k,p,t)
        first,code=judge.evaluate(self.args(),transport=failing)
        self.assertEqual(code,1);self.assertEqual(first['api_calls_made'],16)
        self.assertEqual(first['summary']['expected'],5)
        self.assertEqual(first['summary']['model_failures'],1)
        self.assertEqual(first['summary']['errors'],1)
        self.assertIsNone(first['summary']['joint_qualification']['rate'])
        self.assertTrue(all(v['mean'] is None for v in first['summary']['criteria'].values()))
        self.assertEqual(first['results'][-1]['status'],'model_failure')
        self.assertFalse(first['results'][-1]['joint_qualified'])
        self.assertIsNone(first['results'][0]['dimensions']['OA']['score'])
        second,code=judge.evaluate(self.args(),transport=self.success)
        self.assertEqual(code,0);self.assertEqual(second['api_calls_made'],1)
        self.assertEqual(second['status'],'scored_with_model_failures')
        self.assertTrue(second['summary']['formal_complete'])
        self.assertEqual(second['summary']['joint_qualification']['rate'],.8)
        self.assertEqual(second['summary']['four_dimension_mean'],4)
        self.assertEqual(second['summary']['valid_output_coverage'],.8)
        self.assertEqual(second['summary']['criteria']['IF']['coverage'],.8)
        self.assertEqual(second['summary']['criteria']['IF']['expected'],5)
        self.assertEqual(second['summary']['criteria']['IF']['judge_expected'],4)
        self.assertEqual(second['summary']['criteria']['IF']['judge_coverage'],1)
        Image.new('RGB',(640,480),'blue').save(self.predictions/missing['output_filename'])
        final,code=judge.evaluate(self.args(),transport=self.success)
        self.assertEqual(code,0);self.assertEqual(final['api_calls_made'],4)
        self.assertEqual(final['summary']['joint_qualification']['rate'],1)

    def test_message_template_revision_isolates_scores_without_erasing_paid_history(self):
        args=self.args('--ids',self.tasks[0]['id'])
        first,_=judge.evaluate(args,transport=self.success)
        with patch.object(asset,'MESSAGE_TEMPLATE_REVISION','previous-template-fixture'):
            changed,_=judge.evaluate(args,transport=self.success)
        self.assertEqual(changed['api_calls_made'],4)
        self.assertNotEqual(first['results'][0]['fingerprint'],changed['results'][0]['fingerprint'])
        self.assertFalse(judge.validate_report_protocol(changed['protocol'],self.tasks))
        restored,_=judge.evaluate(args,transport=lambda *a:self.fail('Current paid namespace was erased'))
        self.assertEqual(restored['api_calls_made'],0)

    def test_bad_output_is_model_failure_and_full_image_judge_needs_no_mask_or_references(self):
        path=self.predictions/self.tasks[0]['output_filename'];path.write_bytes(b'not an image')
        report,code=judge.evaluate(self.args('--ids',self.tasks[0]['id']),transport=lambda *a:self.fail('Bad output called API'))
        self.assertEqual(code,0);self.assertEqual(report['summary']['model_failures'],1)
        self.assertEqual(report['summary']['joint_qualification']['rate'],0)
        self.assertIsNone(report['summary']['four_dimension_mean'])
        Image.new('RGB',(640,480),'blue').save(path)
        mask=self.root/'cases'/self.tasks[0]['id']/'allowed_edit_mask.png'
        mask.unlink()
        (self.root/'references.jsonl').unlink()
        report,code=judge.evaluate(self.args('--ids',self.tasks[0]['id']),transport=self.success)
        self.assertEqual(code,0);self.assertEqual(report['summary']['model_failures'],0)
        self.assertTrue(report['summary']['formal_complete'])
        self.assertEqual(report['api_calls_made'],4)

    def test_missing_api_and_dry_run_never_publish_formal_asset_metrics(self):
        with patch.dict(os.environ,{},clear=True):
            report,code=judge.evaluate(self.args('--require-scoring'),transport=lambda *a:self.fail('Missing API made request'))
        self.assertEqual(code,2);self.assertEqual(report['pending_reason'],'needs_api')
        self.assertIsNone(report['summary']['joint_qualification']['rate'])
        report,code=judge.evaluate(self.args('--dry-run'),transport=lambda *a:self.fail('Dry-run made request'))
        self.assertEqual(code,0);self.assertFalse(report['summary']['formal_complete'])
        self.assertIsNone(report['summary']['four_dimension_mean'])

    def test_manifest_provenance_errors_block_formal_summary_instead_of_becoming_model_failures(self):
        cid=self.tasks[0]['id'];manifest=self.root/'generation.json'
        manifest.write_text(json.dumps({'records':[{'id':cid,'status':'complete'}]}))
        args=judge.parser().parse_args(['--dataset',str(self.root),'--generation-manifest',str(manifest),
            '--output',str(self.root/'scores.json'),'--ids',cid,'--no-progress'])
        for error in ('Source hash does not match generation provenance',
                      'Instruction changed since generation; stale output rejected',
                      'Only recorded native top-left ROI is supported',
                      'Unsupported generation adapter_version','Raw output hash mismatch'):
            with self.subTest(error=error),patch('output_adapter.load_prediction',side_effect=ValueError(error)):
                report,code=judge.evaluate(args,transport=lambda *a:self.fail('Invalid provenance made API call'))
            self.assertEqual(code,1);self.assertEqual(report['summary']['model_failures'],0)
            self.assertFalse(report['summary']['formal_complete'])
            self.assertIsNone(report['summary']['joint_qualification']['rate'])
            self.assertEqual(report['results'][0]['failure_type'],'evaluation_input')
        manifest.write_text(json.dumps({'records':[{'id':cid,'status':'error','error':'unsafe provider body'}]}))
        with patch('output_adapter.load_prediction',side_effect=AssertionError('Failed generation was loaded')):
            report,code=judge.evaluate(args,transport=lambda *a:self.fail('Generation failure made API call'))
        self.assertEqual(code,0);self.assertEqual(report['summary']['model_failures'],1)
        self.assertEqual(report['summary']['joint_qualification']['rate'],0)
        self.assertNotIn('unsafe provider body',json.dumps(report))

    def test_wrong_profile_unknown_protocol_mixed_tracks_reference_and_reasoning_config_rejected(self):
        output=self.root/'scores.json';output.write_text('keep')
        original=deepcopy(self.tasks)
        for mutation in ('missing_protocol','unknown_protocol','bad_operation','mixed'):
            self.tasks=deepcopy(original)
            if mutation=='missing_protocol':self.tasks[0].pop('input_protocol')
            elif mutation=='unknown_protocol':self.tasks[0]['input_protocol']='asset-edit-single-source-v99'
            elif mutation=='bad_operation':self.tasks[0]['operation']='color'
            else:self.tasks.append({'id':'legacy_text','operation':'delete','instruction':"Delete 'A'.",'image':original[0]['image']})
            self.write_tasks()
            with self.subTest(mutation=mutation),self.assertRaises(judge.JudgeError):
                judge.evaluate(self.args(),transport=lambda *a:self.fail('Invalid profile made request'))
            self.assertEqual(output.read_text(),'keep')
        self.tasks=original;self.write_tasks()
        with self.assertRaises(judge.JudgeError):judge.evaluate(self.args('--mode','reference'))
        config=self.root/'judge.json';config.write_text(json.dumps({'request_profile':'reasoning_chat'}))
        with self.assertRaises(judge.JudgeError):judge.evaluate(self.args('--config',str(config)))
        self.assertEqual(output.read_text(),'keep')

    def test_summary_joint_threshold_is_per_dimension_and_operation_denominators_are_fixed(self):
        records=[]
        for index,operation in enumerate(asset.METHODS):
            records.append({'id':str(index),'operation':operation,'generation_status':'valid','status':'scored',
                'dimensions':{d:{'status':'scored','score':5 if index else (3 if d=='IF' else 5)}
                              for d in asset.DIMENSIONS}})
        report=judge.summarize(records,'asset')
        self.assertEqual(report['joint_qualification']['rate'],.8)
        self.assertEqual(report['by_operation']['delete']['joint_qualification']['rate'],0)
        self.assertEqual(report['by_operation']['delete']['four_dimension_mean'],4.5)
        records[1]['status']='partial_error';records[1]['dimensions']['OA']={'status':'error','score':None}
        incomplete=judge.summarize(records,'asset')
        self.assertIsNone(incomplete['by_operation']['delete']['four_dimension_mean'])
        self.assertIsNone(incomplete['joint_qualification']['rate'])

    def test_profile_protocol_binding_rejects_stale_prompts_views_and_dimensions(self):
        report,_=judge.evaluate(self.args('--dry-run'),transport=lambda *a:self.fail('Dry run API'))
        self.assertTrue(judge.validate_report_protocol(report['protocol'],self.tasks))
        for key,value in [('version','wrong'),('prompts_sha256','0'*64),('criteria',['IF','TA','VC','LP']),
                          ('view_parameters',{}),('reference_target_supplied',True)]:
            protocol=deepcopy(report['protocol']);protocol[key]=value
            self.assertFalse(judge.validate_report_protocol(protocol,self.tasks))


if __name__=='__main__':unittest.main()
