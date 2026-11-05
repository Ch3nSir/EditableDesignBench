"""Strict text prompts keep legacy scores isolated and original sources intact."""
import json,os,unittest
from unittest.mock import patch
import mllm_evaluate as judge
import text_judge_profile as text
from formal_report import _judge_comparison_protocol
import mllm_test as fixtures

class StrictTextJudgeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AdapterTests();self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        env=patch.dict(os.environ,{'MOCK_JUDGE_KEY':'OFFLINE_ONLY'},clear=True)
        env.start();self.addCleanup(env.stop)

    def test_default_uses_strict_prompts_four_calls_two_native_images(self):
        payloads=[]
        def transport(*values):payloads.append(values[2]);return self.fixture.success(*values)
        report,code=judge.evaluate(self.fixture.args(),transport=transport)
        self.assertEqual(code,0);self.assertEqual(len(payloads),4)
        self.assertEqual(report['protocol']['version'],text.VERSION)
        self.assertEqual(report['protocol']['text_profile'],'strict')
        self.assertTrue(judge.validate_report_protocol(report['protocol'],[self.fixture.task]))
        for p in payloads:
            self.assertIn(p['messages'][0]['content'],text.PROMPTS.values())
            self.assertEqual(p['temperature'],0);self.assertEqual(p['max_tokens'],500)
            self.assertEqual(sum(x['type']=='image_url' for x in p['messages'][1]['content']),2)

    def test_switching_profiles_preserves_and_separates_paid_history(self):
        original=self.fixture.args('--text-profile','texteditbench')
        old,code=judge.evaluate(original,transport=self.fixture.success)
        self.assertEqual(code,0);self.assertEqual(old['protocol']['version'],judge.PROTOCOL_VERSION)
        strict,code=judge.evaluate(self.fixture.args(),transport=self.fixture.success)
        self.assertEqual(code,0);self.assertEqual(strict['api_calls_made'],4)
        self.assertNotEqual(old['results'][0]['fingerprint'],strict['results'][0]['fingerprint'])
        restored,code=judge.evaluate(original,transport=lambda *a:self.fail('Original paid scores were erased'))
        self.assertEqual(code,0);self.assertEqual(restored['api_calls_made'],0)
        self.assertTrue(judge.validate_report_protocol(old['protocol'],[self.fixture.task]))
        strict_cached,code=judge.evaluate(self.fixture.args(),transport=lambda *a:self.fail('Strict paid scores were erased'))
        self.assertEqual(code,0);self.assertEqual(strict_cached['api_calls_made'],0)

    def test_prompt_revision_invalidates_only_its_own_namespace(self):
        first,_=judge.evaluate(self.fixture.args(),transport=self.fixture.success)
        with patch.object(text,'MESSAGE_TEMPLATE_REVISION','other-strict-fixture'):
            changed,_=judge.evaluate(self.fixture.args(),transport=self.fixture.success)
        self.assertEqual(changed['api_calls_made'],4)
        self.assertNotEqual(first['results'][0]['fingerprint'],changed['results'][0]['fingerprint'])
        self.assertNotEqual(_judge_comparison_protocol(first),_judge_comparison_protocol(changed))
        self.assertFalse(judge.validate_report_protocol(changed['protocol'],[self.fixture.task]))

    def test_unknown_profile_fails_before_transport(self):
        with self.assertRaises(judge.JudgeError):judge.current_profile_metadata([self.fixture.task],text_prompt_profile='invalid')

if __name__=='__main__':unittest.main()
