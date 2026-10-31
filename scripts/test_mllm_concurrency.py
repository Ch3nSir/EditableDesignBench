"""Bounded judge concurrency, durable paid scores and visible progress; offline."""
from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Barrier, Lock
import time
import unittest
from unittest.mock import patch

from PIL import Image

import mllm_evaluate as judge


class JudgeConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.predictions=self.root/'predictions';self.predictions.mkdir()
        self.tasks=[]
        for index in range(12):
            cid=f'case{index}';source=self.root/'cases'/cid;source.mkdir(parents=True)
            Image.new('RGB',(16,16),'white').save(source/'source.png')
            Image.new('RGB',(16,16),'blue').save(self.predictions/(cid+'.png'))
            self.tasks.append({'id':cid,'image':f'cases/{cid}/source.png','operation':'delete',
                               'instruction':f"Delete '{cid}'."})
        (self.root/'tasks.jsonl').write_text(''.join(json.dumps(task)+'\n' for task in self.tasks))
        self.environment=patch.dict(os.environ,{'OPENAI_API_KEY':'OFFLINE_FIXTURE',
            'OPENAI_API_URL':'https://judge.invalid/v1','OPENAI_MODEL':'offline-judge'},clear=True)
        self.environment.start();self.addCleanup(self.environment.stop)

    def args(self,workers=5,*extra):
        return judge.parser().parse_args(['--dataset',str(self.root),'--predictions',str(self.predictions),
            '--output',str(self.root/'scores.json'),'--workers',str(workers),'--no-progress',*extra])

    @staticmethod
    def identity(payload):
        dimension=next(d for d in judge.STANDARD_DIMENSIONS if '{"'+d+'": X' in payload['messages'][0]['content'])
        case=int(re.search(r'case(\d+)',payload['messages'][1]['content'][0]['text']).group(1))
        return case,dimension

    @classmethod
    def response(cls,payload):
        case,dimension=cls.identity(payload)
        return {'choices':[{'message':{'content':json.dumps({dimension:3+case%2,'rationale':f'Case {case}.'})},
                            'finish_reason':'stop'}]}

    def test_five_and_ten_workers_overlap_requests_and_preserve_case_identity(self):
        for workers in (5,10):
            with self.subTest(workers=workers):
                lock=Lock();barrier=Barrier(workers);state={'active':0,'peak':0,'calls':0}
                def transport(endpoint,key,payload,timeout):
                    self.assertEqual(payload['temperature'],0)
                    self.assertEqual(payload['max_tokens'],500)
                    with lock:
                        state['active']+=1;state['peak']=max(state['peak'],state['active'])
                        state['calls']+=1;number=state['calls']
                    try:
                        if number<=workers:barrier.wait(timeout=5)
                        time.sleep(.005*(1+self.identity(payload)[0]%3))
                        return self.response(payload)
                    finally:
                        with lock:state['active']-=1
                args=self.args(workers,'--output',str(self.root/f'scores{workers}.json'))
                report,code=judge.evaluate(args,transport=transport)
                self.assertEqual(code,0);self.assertEqual(state['peak'],workers)
                self.assertEqual(state['calls'],48);self.assertEqual(report['api_calls_made'],48)
                self.assertEqual([row['id'] for row in report['results']],[task['id'] for task in self.tasks])
                for index,row in enumerate(report['results']):
                    for dimension in judge.STANDARD_DIMENSIONS:
                        self.assertEqual(row['dimensions'][dimension]['score'],3+index%2)
                args.workers=1
                resumed,code=judge.evaluate(args,transport=lambda *a:self.fail('Changing workers repeated paid calls'))
                self.assertEqual(code,0);self.assertEqual(resumed['api_calls_made'],0)
                self.assertEqual(report['protocol'],resumed['protocol'])

    def test_concurrent_retries_failures_and_missing_inputs_keep_exact_denominators(self):
        (self.predictions/'case7.png').unlink()
        lock=Lock();attempts=Counter()
        def transport(endpoint,key,payload,timeout):
            identity=self.identity(payload)
            self.assertNotEqual(identity[0],7)
            with lock:attempts[identity]+=1;attempt=attempts[identity]
            if identity==(0,'IF') and attempt==1:raise judge.JudgeError('API HTTP 429')
            if identity==(3,'TA'):return {'choices':[{'message':{'content':'invalid JSON'}}]}
            return self.response(payload)
        report,code=judge.evaluate(self.args(5,'--max-retries','1'),transport=transport,sleeper=lambda _:None)
        self.assertEqual(code,1);self.assertEqual(report['api_calls_made'],46)
        self.assertEqual(report['summary']['expected'],12)
        self.assertEqual(report['summary']['scored'],10);self.assertEqual(report['summary']['errors'],2)
        self.assertEqual(report['results'][3]['dimensions']['TA']['status'],'error')
        self.assertIsNone(report['results'][3]['dimensions']['TA']['score'])
        self.assertEqual(report['results'][7]['status'],'error')
        self.assertIsNone(report['summary']['criteria']['TA']['mean'])

    def test_interrupt_checkpoints_every_inflight_paid_response_and_resume_does_not_repeat(self):
        lock=Lock();barrier=Barrier(5);paid=[];interrupted=False;real_write=judge.write_json
        def transport(endpoint,key,payload,timeout):
            with lock:paid.append(self.identity(payload));number=len(paid)
            if number<=5:barrier.wait(timeout=5)
            time.sleep(.01)
            return self.response(payload)
        def interrupt_once(path,value):
            nonlocal interrupted
            if path==self.root/'scores.json' and not interrupted and any(
                dim.get('status')=='scored' for row in value.get('results',[]) for dim in row['dimensions'].values()):
                interrupted=True;raise KeyboardInterrupt()
            return real_write(path,value)
        with patch.object(judge,'write_json',side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):judge.evaluate(self.args(),transport=transport)
        first_paid=list(paid);cache=json.loads(judge.cache_path(self.root/'scores.json').read_text())
        cached={(int(row['id'][4:]),dimension) for row in cache['records'] for dimension in row['dimensions']}
        self.assertEqual(cached,set(first_paid))
        def resumed_transport(endpoint,key,payload,timeout):
            paid.append(self.identity(payload));return self.response(payload)
        report,code=judge.evaluate(self.args(10),transport=resumed_transport)
        self.assertEqual(code,0);self.assertEqual(report['summary']['scored'],12)
        self.assertEqual(report['api_calls_made'],48-len(first_paid))
        self.assertEqual(len(paid),48);self.assertTrue(all(count==1 for count in Counter(paid).values()))

    def test_progress_shows_dimensions_cases_and_cache_and_can_be_disabled(self):
        terminal=io.StringIO()
        with contextlib.redirect_stderr(terminal):
            report,code=judge.evaluate(self.args(5,'--progress'),transport=lambda e,k,p,t:self.response(p))
        self.assertEqual(code,0)
        for value in ('MLLM judge (5 workers)','48/48','cases=12/12','failed=0'):
            self.assertIn(value,terminal.getvalue())
        terminal=io.StringIO()
        with contextlib.redirect_stderr(terminal):
            judge.evaluate(self.args(10,'--progress'),transport=lambda *a:self.fail('Cache made API calls'))
        self.assertIn('cached=48',terminal.getvalue());self.assertIn('api_calls=0',terminal.getvalue())
        terminal=io.StringIO()
        with contextlib.redirect_stderr(terminal):judge.evaluate(self.args(),transport=lambda *a:self.fail('Cache made API calls'))
        self.assertEqual(terminal.getvalue(),'')

    def test_invalid_worker_count_starts_no_requests_and_preserves_previous_output(self):
        output=self.root/'scores.json';output.write_text('unchanged')
        for workers in (0,-1,33):
            with self.subTest(workers=workers),self.assertRaises(judge.JudgeError):
                judge.evaluate(self.args(workers),transport=lambda *a:self.fail('Invalid workers made API call'))
        self.assertEqual(output.read_text(),'unchanged')


if __name__=='__main__':unittest.main()
