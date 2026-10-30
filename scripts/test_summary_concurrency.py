"""Legacy smoke summaries merge safely; formal reports stay per run."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from evaluation_binding import dataset_identity,write_json
import smoke_report


class SummaryConcurrencyTests(unittest.TestCase):
    def test_parallel_smoke_updates_preserve_each_run_and_dataset_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);summary=root/'summary.json'
            summary.write_text(json.dumps({'version':'fixture','ground_truth_revision':2,'tasks':400}))
            (root/'tasks.jsonl').write_text('{"id":"a"}\n')
            before=dataset_identity(root)
            def slower(path,value):
                time.sleep(.01)
                write_json(path,value)
            def update(index):
                run=root/f'model_{index}';run.mkdir()
                smoke_report._update_dataset_summary(summary,run,True,
                    {'mllm_status':'not_scored','pixel_metric_protocol':'texteditbench-official-pixels-v2'},
                    {'records':[]},[{'raw_output':'historical.png'}],
                    {'summary':{'metric_coverage':{'ssim':{'valid':1}}}})
                return str(run.resolve())
            with patch.object(smoke_report,'write_json',side_effect=slower):
                with ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(update,range(4)))
            value=json.loads(summary.read_text())
            self.assertEqual(set(value['model_smoke_runs']),set(paths))
            self.assertTrue(all(row['passed'] for row in value['model_smoke_runs'].values()))
            self.assertEqual(value['tasks'],400)
            self.assertEqual(dataset_identity(root),before)
            self.assertEqual(list(root.glob('summary.json.*.tmp')),[])


if __name__=='__main__':unittest.main()
