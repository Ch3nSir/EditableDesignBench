"""A downloaded collection needs one track flag and a reproducible subset."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench


class BenchAssetCLITests(unittest.TestCase):
    def test_track_sample_and_automatic_evaluation_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);data=base/'data';assets=data/'assets';assets.mkdir(parents=True)
            (data/'tasks.jsonl').write_text(json.dumps({'id':'text-a','operation':'delete','image':'a.png','instruction':'Delete it.'})+'\n')
            rows=[]
            for number in range(30):
                for op in ('delete','insert','replace','move','scale'):
                    rows.append({'id':f'asset_{number}_{op}','source_id':str(number),'operation':op,
                        'input_protocol':'asset-edit-single-source-v1','image':f'cases/asset_{number}_{op}/source.png',
                        'instruction':op+' the photo.'})
            (assets/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            out=base/'run';calls=[]
            def run(command):calls.append([str(x) for x in command]);return 0
            with patch.object(bench,'run_command',side_effect=run),\
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                code=bench.main(['generate','--models','qwen-image21','--dataset-root',str(data),
                    '--track','asset','--sample','20','--output-root',str(out),'--preflight'])
            self.assertEqual(code,0)
            experiment=json.loads((out/'experiment.json').read_text())
            self.assertEqual(experiment['dataset_root'],str(assets))
            self.assertEqual(len(experiment['case_ids']),20)
            chosen=[r for r in rows if r['id'] in experiment['case_ids']]
            self.assertEqual(len({r['source_id'] for r in chosen}),20)
            self.assertEqual({op:sum(r['operation']==op for r in chosen) for op in ('delete','insert','replace','move','scale')},dict.fromkeys(('delete','insert','replace','move','scale'),4))
            self.assertEqual(experiment['sampling']['seed'],42)
            protocol=out/'qwen-image21/protocol.json';protocol.parent.mkdir()
            protocol.write_text(json.dumps({'model_id':'qwen-image21','case_ids':experiment['case_ids']}))
            args=bench.parser().parse_args(['evaluate','--runs-root',str(out),'--pixel-only'])
            _,dataset,_,_,expected=bench.evaluation_context(args)
            self.assertEqual(dataset,assets)
            self.assertEqual(set(expected.split(',')),set(experiment['case_ids']))


if __name__=='__main__':unittest.main()
