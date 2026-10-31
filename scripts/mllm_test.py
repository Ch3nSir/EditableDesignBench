"""Offline contract tests for the MLLM adapter. No API/network is used."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import mllm_evaluate as m


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'cases/a').mkdir(parents=True);(self.root/'predictions').mkdir()
        Image.new('RGB',(16,16),'white').save(self.root/'cases/a/source.png')
        Image.new('RGB',(16,16),'blue').save(self.root/'predictions/a.png')
        self.task={'id':'a','source_id':'x','operation':'color','instruction':"Change 'A' to blue.",
                   'image':'cases/a/source.png','output_filename':'a.png'}
        (self.root/'tasks.jsonl').write_text(json.dumps(self.task)+'\n')
        self.config=self.root/'config.json';self.config.write_text(json.dumps({'model':'mock-judge','api_key_env':'MOCK_JUDGE_KEY','base_url':'https://judge.example/v1','max_retries':0}))

    def tearDown(self):self.temp.cleanup()

    def test_luna_environment_only_matches_explicit_config_and_reuses_paid_scores(self):
        args=self.args();args.config=None
        environment={'OPENAI_API_KEY':'offline-fixture','OPENAI_API_URL':'https://judge.example/v1',
                     'OPENAI_MODEL':'gpt-6-luna'}
        config=json.loads((m.ROOT/'config/judge.example.json').read_text())
        self.config.write_text(json.dumps(config));payloads=[]
        def capture(*values):
            payloads.append(values[2])
            return self.success(*values)
        with patch.dict(os.environ,environment,clear=True):
            first,code=m.evaluate(args,transport=capture)
            self.assertEqual(code,0);self.assertEqual(first['api_calls_made'],4)
            self.assertEqual(first['protocol']['temperature'],0)
            self.assertEqual(first['protocol']['max_tokens_per_criterion'],500)
            for payload in payloads:
                self.assertEqual(payload['temperature'],0)
                self.assertEqual(payload['max_tokens'],500)
                self.assertNotIn('max_completion_tokens',payload)
                self.assertNotIn('reasoning_effort',payload)
            self.assertEqual(m.request_parameters(config),{'temperature':0.0,'max_tokens':500})
            args.config=self.config
            resumed,code=m.evaluate(args,transport=lambda *a:self.fail('Equivalent explicit config repeated API calls'))
            self.assertEqual(code,0);self.assertEqual(resumed['api_calls_made'],0)

    def test_model_name_does_not_override_official_request_parameters(self):
        args=self.args()
        for model in ('gpt-6-luna','gpt-4o','mock-judge'):
            self.config.write_text(json.dumps({'model':model}))
            with self.subTest(model=model),patch.dict(os.environ,{},clear=True):
                self.assertEqual(m.load_credentials(args)['request_parameters'],{'temperature':0.0,'max_tokens':500})

    def test_official_parameters_do_not_reuse_reasoning_profile_scores(self):
        config=json.loads(self.config.read_text());config['request_profile']='reasoning_chat'
        self.config.write_text(json.dumps(config))
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'offline-fixture'},clear=True):
            first,code=m.evaluate(self.args(),transport=self.success)
            self.assertEqual(code,0)
            config['request_profile']='texteditbench';self.config.write_text(json.dumps(config))
            second,code=m.evaluate(self.args(),transport=self.success)
            self.assertEqual(code,0);self.assertEqual(second['api_calls_made'],4)
            self.assertNotEqual(first['results'][0]['fingerprint'],second['results'][0]['fingerprint'])

    def test_reasoning_parameters_and_usage_are_preserved(self):
        config=json.loads(self.config.read_text())
        config.update(request_profile='reasoning_chat',max_completion_tokens=4096,
                      reasoning_effort='low',store=False)
        self.config.write_text(json.dumps(config));payloads=[]
        def capture(endpoint,key,payload,timeout):
            payloads.append(payload)
            result=self.success(endpoint,key,payload,timeout)
            return dict(result,model='mock-judge',usage={'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,'secret_field':'excluded'})
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            report,code=m.evaluate(self.args(),transport=capture)
        self.assertEqual(code,0);self.assertEqual(len(payloads),4)
        for payload in payloads:
            self.assertNotIn('temperature',payload);self.assertNotIn('max_tokens',payload)
            self.assertEqual(payload['max_completion_tokens'],4096)
            self.assertEqual(payload['reasoning_effort'],'low');self.assertIs(payload['store'],False)
        self.assertIsNone(report['protocol']['temperature'])
        self.assertEqual(report['protocol']['max_tokens_per_criterion'],4096)
        self.assertEqual(report['protocol']['request_parameters'],m.request_parameters(config))
        score=report['results'][0]['dimensions']['IF']
        self.assertEqual(score['usage'],{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110})
        self.assertEqual(score['returned_model'],'mock-judge')

    def test_reasoning_effort_changes_invalidate_paid_cache_and_comparison(self):
        from formal_report import _judge_comparison_protocol
        config=json.loads(self.config.read_text())
        config.update(request_profile='reasoning_chat',reasoning_effort='low')
        self.config.write_text(json.dumps(config))
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            first,_=m.evaluate(self.args(),transport=self.success)
            resumed,_=m.evaluate(self.args(),transport=lambda *a:self.fail('Paid score repeated'))
            config['reasoning_effort']='medium';self.config.write_text(json.dumps(config))
            second,_=m.evaluate(self.args(),transport=self.success)
        self.assertEqual(first['api_calls_made'],4);self.assertEqual(resumed['api_calls_made'],0)
        self.assertEqual(second['api_calls_made'],4)
        self.assertNotEqual(_judge_comparison_protocol(first),_judge_comparison_protocol(second))

    def test_invalid_request_parameters_fail_before_api(self):
        self.assertEqual(m.request_parameters({}),{'temperature':0.0,'max_tokens':500})
        original=json.loads(self.config.read_text())
        for bad in ({'request_profile':'unknown'},{'store':'false'},
                    {'request_profile':'reasoning_chat','max_completion_tokens':True},
                    {'request_profile':'reasoning_chat','max_completion_tokens':0},
                    {'request_profile':'reasoning_chat','max_completion_tokens':128001},
                    {'request_profile':'reasoning_chat','reasoning_effort':'typo'}):
            with self.subTest(config=bad):
                config=dict(original);config.update(bad)
                self.config.write_text(json.dumps(config))
                with self.assertRaises(m.JudgeError):
                    m.evaluate(self.args(),transport=lambda *a:self.fail('Invalid config made API call'))

    def args(self,*extra):
        return m.parser().parse_args(['--dataset',str(self.root),'--predictions',str(self.root/'predictions'),
            '--output',str(self.root/'scores.json'),'--config',str(self.config),
            '--workers','1','--no-progress',*extra])

    @staticmethod
    def success(endpoint,key,payload,timeout):
        prompt=payload['messages'][0]['content']
        dim=next(d for d in ('IF','TA','VC','LP','SE') if '{"'+d+'": X' in prompt)
        return {'choices':[{'message':{'content':json.dumps({dim:4,'rationale':'Checked against input and instruction.'})},'finish_reason':'stop'}]}

    def test_prompts_are_exact_snapshots(self):
        data=json.loads(m.PROMPT_FILE.read_text())
        for item in data['modes'].values():
            p=m.resolve_prompt_source(item);tree=ast.parse(p.read_text(encoding='utf-8-sig'))
            node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
            self.assertEqual(item['prompts'],ast.literal_eval(node.value))
            self.assertEqual(item['source_sha256'],m.digest(p.read_bytes()))

    def test_complete_requests_match_official_both_modes_all_dimensions(self):
        """Execute only the official request function with a local stub client.

        No official module imports, configuration, credentials, or network run.
        Use identical sentinel image URLs to compare all message text, ordering,
        editing-method rules, optional SE content, and generation parameters.
        Production encoding remains validated lossless RGB PNG in our adapter.
        """
        modes=json.loads(m.PROMPT_FILE.read_text())['modes']
        for mode,item in modes.items():
            source=m.resolve_prompt_source(item)
            tree=ast.parse(source.read_text(encoding='utf-8-sig'))
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                          and n.name=='call_llm_evaluation')
            module=ast.Module(body=[function],type_ignores=[])
            official_requests=[]
            def official_create(**payload):
                official_requests.append(payload)
                dimension=next(d for d,p in item['prompts'].items()
                               if p==payload['messages'][0]['content'])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({dimension:4,'rationale':'Offline contract response.'})))])
            namespace={'PROMPTS':item['prompts'],'API_MODEL':'mock-judge','json':json,
                       'logger':Mock(),'CLIENT':SimpleNamespace(chat=SimpleNamespace(
                           completions=SimpleNamespace(create=official_create)))}
            exec(compile(module,str(source),'exec'),namespace)
            for operation,method in m.METHODS.items():
                for dimension in (*m.STANDARD_DIMENSIONS,'SE'):
                    with self.subTest(mode=mode,operation=operation,dimension=dimension):
                        task=dict(self.task,operation=operation,
                                  instruction="Replace '春天' with 'summer'.",
                                  knowledge_prompt='Use the season after spring.')
                        kwargs={'dimension':dimension,'input_b64':'SOURCE','pred_b64':'PRED',
                                'prompt':task['instruction'],'retry_count':1}
                        if dimension=='IF' or mode=='reference':kwargs['editing_method']=method
                        if dimension=='SE':kwargs['knowledge_prompt']=task['knowledge_prompt']
                        if mode=='reference':kwargs['output_b64']='REFERENCE'
                        result=namespace['call_llm_evaluation'](**kwargs)
                        self.assertEqual(result[dimension],4)
                        expected=official_requests[-1]
                        messages=m.build_messages(dimension,item['prompts'][dimension],task,
                            'data:image/jpeg;base64,SOURCE','data:image/jpeg;base64,PRED',
                            'data:image/jpeg;base64,REFERENCE' if mode=='reference' else None)
                        self.assertEqual(messages,expected['messages'])
                        ours=[]
                        def capture(endpoint,key,payload,timeout):
                            ours.append(payload)
                            return {'choices':[{'message':{'content':json.dumps({
                                dimension:4,'rationale':'Offline contract response.'})},
                                'finish_reason':'stop'}]}
                        credentials={'endpoint':'https://judge.example/v1/chat/completions',
                                     'api_key':'offline','model':'mock-judge','timeout':1,'max_retries':0}
                        self.assertEqual(m.request_score(dimension,messages,credentials,capture)['status'],'scored')
                        self.assertEqual(ours,[expected])

    def test_no_reference_dry_run_never_calls_transport(self):
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            report,code=m.evaluate(self.args('--dry-run'),transport=lambda *a:self.fail('API called'))
        self.assertEqual(code,0);self.assertEqual(report['status'],'not_scored')
        self.assertEqual(report['summary']['criteria']['IF']['mean'],None)
        self.assertEqual(report['results'][0]['dimensions']['SE']['status'],'not_applicable')
        self.assertNotIn('secret',(self.root/'scores.json').read_text())

    def test_missing_api_pending_and_explicit_strict_mode(self):
        with patch.dict(os.environ,{},clear=True):
            report,code=m.evaluate(self.args('--require-scoring'),transport=lambda *a:self.fail('API called'))
        self.assertEqual(code,2);self.assertEqual(report['pending_reason'],'needs_api')
        self.assertTrue(all(v['score'] is None for v in report['results'][0]['dimensions'].values()))

    def test_message_modes_do_not_leak_reference(self):
        msg=m.build_messages('IF','official',self.task,'source','prediction')
        images=[x['image_url']['url'] for x in msg[1]['content'] if x['type']=='image_url']
        self.assertEqual(images,['source','prediction'])
        msg=m.build_messages('IF','official',self.task,'source','prediction','reference')
        self.assertEqual([x['image_url']['url'] for x in msg[1]['content'] if x['type']=='image_url'],['source','prediction','reference'])

    def test_invalid_score_never_becomes_fake_zero(self):
        for score in [True,-1,6,float('nan'),'4']:
            with self.assertRaises(m.JudgeError):m.parse_score(json.dumps({'IF':score,'rationale':'x'}),'IF')
        creds={'endpoint':'https://judge.example/v1/chat/completions','api_key':'x','model':'x','timeout':1,'max_retries':1}
        result=m.request_score('IF',[],creds,lambda *a:{'choices':[{'message':{'content':'bad'}}]},lambda _:None)
        self.assertEqual(result['status'],'error');self.assertIsNone(result['score']);self.assertEqual(result['attempts'],2)

    def test_resume_uses_input_and_prediction_hashes(self):
        calls=[]
        def mocked(*args):calls.append(1);return self.success(*args)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            r,c=m.evaluate(self.args(),mocked,lambda _:None);self.assertEqual(c,0)
            self.assertEqual(len(calls),4)
            m.evaluate(self.args(),mocked,lambda _:None);self.assertEqual(len(calls),4)
            Image.new('RGB',(16,16),'red').save(self.root/'predictions/a.png')
            m.evaluate(self.args(),mocked,lambda _:None);self.assertEqual(len(calls),8)
        self.assertEqual(r['summary']['scored'],1)

    def test_request_protocol_change_invalidates_cached_scores(self):
        calls=[]
        def mocked(*args):calls.append(1);return self.success(*args)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            with patch.object(m,'PROTOCOL_VERSION','texteditbench-mllm-adapter-v1'):
                m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(len(calls),4)
            report,code=m.evaluate(self.args(),mocked,lambda _:None)
        self.assertEqual(code,0)
        self.assertEqual(len(calls),8)
        self.assertEqual(report['protocol']['version'],'texteditbench-mllm-adapter-v2')

    def test_interrupt_before_validation_does_not_erase_paid_cached_scores(self):
        calls=[]
        def mocked(*args):calls.append(1);return self.success(*args)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(len(calls),4)
            with patch.object(m,'prepare_case',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):m.evaluate(self.args(),mocked,lambda _:None)
            pending=json.loads(m.cache_path(self.root/'scores.json').read_text())
            self.assertEqual(len(pending['records']),1)
            self.assertEqual(len(pending['records'][0]['dimensions']),4)
            report,code=m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(code,0)
            self.assertEqual(len(calls),4)
            self.assertNotIn('_resume_cache',report)

    def add_second_task(self):
        task=dict(self.task,id='b',output_filename='b.png')
        (self.root/'tasks.jsonl').write_text(json.dumps(self.task)+'\n'+json.dumps(task)+'\n')
        Image.new('RGB',(16,16),'blue').save(self.root/'predictions/b.png')

    def invoke_main(self,*extra):
        args=self.args(*extra)
        command=['mllm_evaluate.py','--dataset',str(args.dataset),'--predictions',str(args.predictions),
                 '--output',str(args.output),'--config',str(args.config),*extra]
        with patch.object(sys,'argv',command),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return m.main()

    def test_invalid_request_or_configuration_preserves_existing_report_and_cache(self):
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),self.success,lambda _:None)
        output=self.root/'scores.json';cache=m.cache_path(output)
        before=(output.read_bytes(),cache.read_bytes())
        config_before=self.config.read_bytes()
        for failure in ('id','config','manifest','corrupt_report'):
            with self.subTest(failure=failure):
                extra=[]
                if failure=='id':extra=['--ids','unknown']
                if failure=='config':self.config.write_text('{invalid')
                if failure=='manifest':
                    extra=['--generation-manifest',str(self.root/'missing-manifest.json')]
                    # Mutually exclusive CLI options are validated by argparse;
                    # use direct evaluate + main's error handler for this case.
                    args=self.args();args.predictions=None;args.generation_manifest=self.root/'missing-manifest.json'
                if failure=='corrupt_report':output.write_text('{corrupted')
                preserved=(output.read_bytes(),cache.read_bytes())
                with patch.object(m,'post_json',side_effect=AssertionError('API called')):
                    if failure=='manifest':
                        with patch.object(m,'parser') as parser:
                            parser.return_value.parse_args.return_value=args
                            with contextlib.redirect_stderr(io.StringIO()):code=m.main()
                    else:code=self.invoke_main(*extra)
                self.assertEqual(code,1)
                self.assertEqual((output.read_bytes(),cache.read_bytes()),preserved)
                self.assertTrue(json.loads((self.root/'scores.json.error.json').read_text())['evaluation_output_preserved'])
                self.config.write_bytes(config_before);output.write_bytes(before[0])

    def test_temporary_prediction_failure_retains_paid_cache_without_reusing_invalid_input(self):
        calls=[]
        def mocked(*args):calls.append(1);return self.success(*args)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),mocked,lambda _:None)
            before=(self.root/'predictions/a.png').read_bytes()
            (self.root/'predictions/a.png').unlink()
            failure,code=m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(code,1)
            self.assertIsNone(failure['summary']['criteria']['IF']['mean'])
            self.assertEqual(failure['summary']['criteria']['IF']['coverage'],0)
            cached=json.loads(m.cache_path(self.root/'scores.json').read_text())
            self.assertEqual(len(cached['records'][0]['dimensions']),4)
            (self.root/'predictions/a.png').write_bytes(before)
            report,code=m.evaluate(self.args(),mocked,lambda _:None)
        self.assertEqual(code,0);self.assertEqual(len(calls),4)
        self.assertEqual(report['summary']['scored'],1)

    def test_subset_keeps_other_cases_and_changed_fingerprints_in_separate_cache(self):
        self.add_second_task();calls=[]
        def mocked(*args):calls.append(1);return self.success(*args)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(len(calls),8)
            subset,code=m.evaluate(self.args('--ids','a'),mocked,lambda _:None)
            self.assertEqual(code,0);self.assertEqual(subset['summary']['expected'],1)
            self.assertEqual([r['id'] for r in subset['results']],['a'])
            self.assertEqual({r['id'] for r in json.loads(m.cache_path(self.root/'scores.json').read_text())['records']},{'a','b'})
            _,code=m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(code,0);self.assertEqual(len(calls),8)
            old=(self.root/'predictions/a.png').read_bytes()
            Image.new('RGB',(16,16),'red').save(self.root/'predictions/a.png')
            m.evaluate(self.args('--ids','a'),mocked,lambda _:None)
            self.assertEqual(len(calls),12)
            (self.root/'predictions/a.png').write_bytes(old)
            m.evaluate(self.args(),mocked,lambda _:None)
        self.assertEqual(len(calls),12)

    def test_paid_dimension_survives_interruption_before_report_write(self):
        calls=[];real_write=m.write_json
        def mocked(*args):calls.append(1);return self.success(*args)
        def interrupt_after_cached_score(path,obj):
            if path==self.root/'scores.json' and any(r.get('dimensions',{}).get('IF',{}).get('status')=='scored' for r in obj.get('results',[])):
                raise KeyboardInterrupt()
            return real_write(path,obj)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            with patch.object(m,'write_json',side_effect=interrupt_after_cached_score):
                with self.assertRaises(KeyboardInterrupt):m.evaluate(self.args(),mocked,lambda _:None)
            self.assertEqual(len(calls),1)
            report,code=m.evaluate(self.args(),mocked,lambda _:None)
        self.assertEqual(code,0);self.assertEqual(len(calls),4)
        self.assertEqual(report['summary']['scored'],1)

    def test_newer_cached_paid_response_wins_interrupted_older_report(self):
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),self.success,lambda _:None)
            output=self.root/'scores.json';old_report=output.read_bytes()
            real_write=m.write_json
            def upgraded(*values):
                result=self.success(*values)
                item=json.loads(result['choices'][0]['message']['content'])
                dimension=next(k for k in item if k!='rationale');item[dimension]=5
                result['choices'][0]['message']['content']=json.dumps(item)
                return result
            def interrupt(path,obj):
                if path==output and any(r.get('dimensions',{}).get('IF',{}).get('score')==5 for r in obj.get('results',[])):
                    raise KeyboardInterrupt()
                return real_write(path,obj)
            with patch.object(m,'write_json',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):m.evaluate(self.args('--no-resume'),upgraded,lambda _:None)
            # Simulate the previous intact report remaining after its replacement
            # was interrupted, while the paid response already reached cache.
            output.write_bytes(old_report)
            report,code=m.evaluate(self.args(),lambda *a:self.fail('Cached score requested again'))
        self.assertEqual(code,0)
        self.assertEqual(report['results'][0]['dimensions']['IF']['score'],5)
        self.assertEqual(report['results'][0]['dimensions']['TA']['score'],4)

    def test_legacy_report_cache_migrates_without_repeating_paid_requests(self):
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            first,code=m.evaluate(self.args(),self.success,lambda _:None)
            output=self.root/'scores.json';m.cache_path(output).unlink()
            legacy={'protocol':first['protocol'],'status':'running','results':[m.unprocessed_record(self.task)],
                    '_resume_cache':first['results']}
            output.write_text(json.dumps(legacy))
            report,code=m.evaluate(self.args(),lambda *a:self.fail('Legacy paid score requested again'))
        self.assertEqual(code,0);self.assertEqual(report['summary']['scored'],1)
        self.assertEqual(len(json.loads(m.cache_path(output).read_text())['records']),1)

    def test_backups_are_recoverable_and_output_lock_blocks_second_writer(self):
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),self.success,lambda _:None)
            output=self.root/'scores.json';sidecar=m.cache_path(output)
            before=(output.read_bytes(),sidecar.read_bytes())
            m.evaluate(self.args('--dry-run'),lambda *a:self.fail('API called'))
            self.assertEqual(output.with_name(output.name+'.bak').read_bytes(),before[0])
            self.assertEqual(sidecar.with_name(sidecar.name+'.bak').read_bytes(),before[1])
            output_before=output.read_bytes();cache_before=sidecar.read_bytes()
            with m.output_lock(output):
                with self.assertRaisesRegex(m.JudgeError,'active evaluation'):
                    m.evaluate(self.args(),lambda *a:self.fail('API called'))
            self.assertEqual((output.read_bytes(),sidecar.read_bytes()),(output_before,cache_before))
            self.assertFalse(list(self.root.glob('*.tmp')))

    def test_judge_modes_models_and_protocols_do_not_reuse_other_paid_scores(self):
        calls=[]
        def mocked(*args):calls.append(1);return self.success(*args)
        Image.new('RGB',(16,16),'white').save(self.root/'cases/a/target.png')
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            m.evaluate(self.args(),mocked,lambda _:None)
            m.evaluate(self.args('--model','another-judge'),mocked,lambda _:None)
            self.assertEqual(len(calls),8)
            m.evaluate(self.args('--mode','reference'),mocked,lambda _:None)
            self.assertEqual(len(calls),12)
            m.evaluate(self.args(),mocked,lambda _:None)
        self.assertEqual(len(calls),12)

    def test_failed_inputs_remain_in_coverage_denominator(self):
        (self.root/'predictions/a.png').unlink()
        with patch.dict(os.environ,{},clear=True):report,code=m.evaluate(self.args())
        self.assertEqual(code,1);self.assertEqual(report['summary']['expected'],1)
        self.assertEqual(report['summary']['criteria']['IF']['expected'],1)
        self.assertEqual(report['summary']['criteria']['IF']['coverage'],0)

    def test_incomplete_criterion_sets_have_no_aggregate_mean(self):
        def record(status,score):
            return {'status':'scored' if status=='scored' else status,
                    'dimensions':{'IF':{'status':status,'score':score}}}
        for status in ['error','not_scored']:
            with self.subTest(status=status):
                summary=m.summarize([record('scored',5),record(status,None)])
                self.assertIsNone(summary['criteria']['IF']['mean'])
                self.assertEqual(summary['criteria']['IF']['coverage'],.5)
                self.assertEqual(summary['criteria']['IF']['expected'],2)
        complete=m.summarize([record('scored',5),record('scored',3)])
        self.assertEqual(complete['criteria']['IF']['mean'],4)
        self.assertEqual(complete['criteria']['IF']['coverage'],1)
        self.assertIsNone(m.summarize([])['criteria']['IF']['mean'])

    def test_intermediate_checkpoints_include_all_requested_cases(self):
        second=dict(self.task,id='b',output_filename='b.png')
        (self.root/'tasks.jsonl').write_text(json.dumps(self.task)+'\n'+json.dumps(second)+'\n')
        Image.new('RGB',(16,16),'blue').save(self.root/'predictions/b.png')
        calls=[]
        def mocked(*args):
            calls.append(1)
            current=json.loads((self.root/'scores.json').read_text())
            self.assertEqual(current['summary']['expected'],2)
            self.assertEqual(current['summary']['criteria']['IF']['expected'],2)
            if len(calls)==5:
                self.assertIsNone(current['summary']['criteria']['IF']['mean'])
                self.assertEqual(current['summary']['criteria']['IF']['coverage'],.5)
            return self.success(*args)
        with patch.dict(os.environ,{'MOCK_JUDGE_KEY':'secret'}):
            report,code=m.evaluate(self.args(),mocked,lambda _:None)
        self.assertEqual(code,0)
        self.assertEqual(report['summary']['criteria']['IF']['mean'],4)

    def test_unknown_ids_fail_before_requests(self):
        with self.assertRaises(m.JudgeError):m.evaluate(self.args('--ids','not-real'),transport=lambda *a:self.fail('API called'))

    def test_duplicate_and_empty_id_tokens_fail_before_requests(self):
        for ids in ['a,a','a,','a,,a','',',a']:
            with self.subTest(ids=ids):
                with self.assertRaises(m.JudgeError):m.evaluate(self.args('--ids',ids),transport=lambda *a:self.fail('API called'))

    def test_padded_manifest_evaluates_native_roi_without_rewriting(self):
        from output_adapter import prepare_rgb_record,sha256_file,instruction_sha256
        raw=self.root/'raw.png';rgb=self.root/'rgb.png'
        Image.new('RGBA',(32,32),(0,0,255,255)).save(raw)
        record=prepare_rgb_record(raw,rgb,{'id':'a','source_size':[16,16],'canvas_size':[32,32],
            'evaluation_roi':[0,0,16,16],'padding':{'left':0,'top':0,'right':16,'bottom':16,'mode':'edge'},
            'source_sha256':sha256_file(self.root/'cases/a/source.png'),'instruction_sha256':instruction_sha256(self.task)})
        manifest=self.root/'generation.json';manifest.write_text(json.dumps({'schema_version':1,'records':[record]}))
        args=m.parser().parse_args(['--dataset',str(self.root),'--generation-manifest',str(manifest),
            '--output',str(self.root/'scores.json'),'--config',str(self.config),'--dry-run'])
        before=(raw.read_bytes(),rgb.read_bytes())
        report,code=m.evaluate(args,lambda *a:self.fail('API called'))
        self.assertEqual(code,0);self.assertEqual(report['results'][0]['prediction_provenance']['evaluation_roi'],[0,0,16,16])
        self.assertEqual(before,(raw.read_bytes(),rgb.read_bytes()))


if __name__=='__main__':unittest.main()
