"""CLI planning tests; every model/API/Slurm subprocess is simulated."""
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import bench


class BenchCLITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.dataset=self.root/'data set';self.dataset.mkdir()
        self.models=bench.model_ids()[:2]
        self.tasks=[{'id':f'case{i:03}','image':f'cases/case{i:03}/source.png',
                     'instruction':f"Delete '{i}'.",'operation':'delete'} for i in range(400)]
        (self.dataset/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))
        self.experiment=self.root/'experiment $(literal) with spaces'
        self.calls=[];self.active=set();self.next_job=100

    def tearDown(self):self.temp.cleanup()

    def invoke(self,*arguments):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return bench.main(list(arguments))

    def local(self,command):
        self.calls.append([str(v) for v in command]);return 0

    def scheduler(self,command,**kwargs):
        self.calls.append([str(v) for v in command])
        if command[0]=='sbatch':
            job=str(self.next_job);self.next_job+=1;self.active.add(job)
            return SimpleNamespace(returncode=0,stdout=job+'\n',stderr='')
        if command[0]=='squeue':
            job=str(command[command.index('-j')+1])
            return SimpleNamespace(returncode=0,stdout='RUNNING\n' if job in self.active else '',stderr='')
        raise AssertionError('Unexpected live subprocess: '+str(command))

    def generate_args(self,*extra):
        return ['generate','--models',','.join(self.models),'--dataset-root',str(self.dataset),
                '--output-root',str(self.experiment),'--all',*extra]

    def plan_local(self):
        with patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke(*self.generate_args('--preflight')),0)
        return json.loads((self.experiment/'experiment.json').read_text())

    def make_protocols(self):
        experiment=self.plan_local()
        for model in self.models:
            run=self.experiment/model;run.mkdir(parents=True,exist_ok=True)
            (run/'protocol.json').write_text(json.dumps({'model_id':model,'run_id':experiment['run_id'],
                                                        'case_ids':experiment['case_ids']}))
        self.calls=[]

    def test_generate_plans_same400_cases_in_distinct_model_directories(self):
        plan=self.plan_local()
        self.assertEqual(len(plan['case_ids']),400);self.assertEqual(plan['models'],self.models)
        self.assertEqual(len(self.calls),2)
        for model,command in zip(self.models,self.calls):
            self.assertEqual(command[command.index('--model')+1],model)
            self.assertEqual(command[command.index('--output-dir')+1],str(self.experiment/model))
            self.assertEqual(len(command[command.index('--ids')+1].split(',')),400)
            self.assertIn('--preflight',command)
        self.assertEqual(len({command[command.index('--output-dir')+1] for command in self.calls}),2)

    def test_custom_config_is_forwarded_to_local_and_slurm_generators_and_downloader(self):
        config=self.root/'extra models.json'
        profile=bench.model_registry.get_model_config('qwen-edit-2511')
        profile.pop('local_dir')
        config.write_text(json.dumps({'models':{'custom-edit':profile}}))
        with patch.dict(bench.MODEL_CONFIGS),patch.object(bench,'run_command',side_effect=self.local):
            args=self.generate_args('--model-config',str(config),'--preflight')
            args[args.index('--models')+1]='custom-edit'
            self.assertEqual(self.invoke(*args),0)
            command=self.calls[-1]
            self.assertEqual(command[command.index('--model-config')+1],str(config.resolve()))
            with patch.object(bench.subprocess,'run',side_effect=self.scheduler):
                args.remove('--preflight');args.append('--submit')
                self.assertEqual(self.invoke(*args),0)
            jobs=json.loads((self.experiment/'jobs.json').read_text())
            script=Path(jobs['entries']['generate:custom-edit']['script']).read_text()
            self.assertIn('--model-config',script);self.assertIn(str(config),script)
            self.assertEqual(self.invoke('download','--models','custom-edit','--model-config',str(config)),0)
            self.assertIn(str(config),self.calls[-1])

    def test_evaluate_and_compare_all_include_optional_and_custom_experiment_models(self):
        self.models=['qwen-edit-2511','custom-api-model']
        with patch.dict(bench.MODEL_CONFIGS,{'custom-api-model':bench.MODEL_CONFIGS['qwen-edit-2511']}):
            self.make_protocols()
        with patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke('evaluate','--runs-root',str(self.experiment),'--pixel-only'),0)
            self.assertEqual(self.invoke('compare','--runs-root',str(self.experiment),'--pixel-only'),0)
        evaluated=[c for c in self.calls if Path(c[1]).name=='evaluate.py']
        self.assertEqual(len(evaluated),2)
        command=self.calls[-1]
        self.assertEqual(command[command.index('--models')+1],','.join(self.models))

    def test_default_all_uses_registry_default_profile_not_optional_models(self):
        defaults=bench.model_ids()
        with patch.object(bench.model_registry,'DEFAULT_MODELS',tuple(defaults[:1]),create=True):
            self.assertEqual(bench.model_ids(),defaults[:1])
            # Evaluation's all includes every model recorded in the experiment.
            self.assertEqual(bench.model_ids('all',defaults),defaults)
        with self.assertRaises(ValueError):bench.model_ids(self.models[0],[self.models[1]])

    def test_experiment_selection_is_immutable(self):
        self.plan_local();before=(self.experiment/'experiment.json').read_bytes()
        arguments=self.generate_args('--preflight');index=arguments.index('--all')
        arguments[index:index+1]=['--ids','case000']
        with patch.object(bench,'run_command',side_effect=self.local):self.assertEqual(self.invoke(*arguments),2)
        self.assertEqual((self.experiment/'experiment.json').read_bytes(),before)

    def test_native_predictions_are_scored_in_one_command_with_the_same_subset(self):
        predictions=self.root/'predictions'/'MyModel';predictions.mkdir(parents=True)
        destination=self.root/'scores'
        with patch.object(bench,'judge_ready'),patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke('evaluate','--predictions',str(predictions),
                '--dataset-root',str(self.dataset),'--ids','case000,case002','--output-dir',str(destination)),0)
        self.assertEqual([Path(c[1]).name for c in self.calls],['evaluate.py','mllm_evaluate.py'])
        for command in self.calls:
            self.assertEqual(command[command.index('--ids')+1],'case000,case002')
            self.assertEqual(command[command.index('--predictions')+1],str(predictions))
            self.assertNotIn('--config',command)
        self.assertIn(str(destination/'pixel_scores.json'),self.calls[0])
        self.assertIn(str(destination/'mllm_scores.json'),self.calls[1])

    def test_native_prediction_failure_still_reports_both_tracks_and_returns_failure(self):
        predictions=self.root/'predictions';predictions.mkdir()
        def run(command):
            self.local(command)
            return 1 if Path(command[1]).name=='evaluate.py' else 0
        with patch.object(bench,'judge_ready'),patch.object(bench,'run_command',side_effect=run):
            code=self.invoke('evaluate','--predictions',str(predictions),'--dataset-root',str(self.dataset),
                '--output-dir',str(self.root/'scores'))
        self.assertEqual(code,1);self.assertEqual(len(self.calls),2)

    def test_native_pixel_only_needs_no_credentials_and_invalid_ids_start_no_processes(self):
        predictions=self.root/'predictions';predictions.mkdir()
        arguments=['evaluate','--predictions',str(predictions),'--dataset-root',str(self.dataset),
                   '--output-dir',str(self.root/'scores'),'--pixel-only']
        with patch.object(bench,'judge_ready',side_effect=AssertionError('Pixel-only asked for an API key')),\
                patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke(*arguments,'--ids','not-a-task'),2)
            self.assertEqual(self.calls,[])
            self.assertEqual(self.invoke(*arguments),0)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(len(self.calls[0][self.calls[0].index('--ids')+1].split(',')),400)

    def test_judge_workers_and_progress_options_reach_local_native_and_slurm_commands(self):
        self.make_protocols()
        args=['evaluate','--runs-root',str(self.experiment),'--dry-run','--workers','10','--no-progress']
        with patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke(*args),0)
        judges=[c for c in self.calls if Path(c[1]).name=='mllm_evaluate.py']
        for command in judges:
            self.assertEqual(command[command.index('--workers')+1],'10');self.assertIn('--no-progress',command)
        with patch.object(bench.subprocess,'run',side_effect=self.scheduler):
            self.assertEqual(self.invoke(*args,'--submit'),0)
        jobs=json.loads((self.experiment/'jobs.json').read_text())
        for model in self.models:
            text=Path(jobs['entries']['judge:'+model]['script']).read_text()
            self.assertIn('--workers 10',text);self.assertIn('--no-progress',text)
        predictions=self.root/'predictions';predictions.mkdir();self.calls=[]
        with patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke('evaluate','--predictions',str(predictions),'--dataset-root',str(self.dataset),
                '--output-dir',str(self.root/'scores'),'--dry-run','--workers','10'),0)
        command=self.calls[-1]
        self.assertEqual(command[command.index('--workers')+1],'10')

    def test_missing_formal_credentials_fail_before_any_process_or_jobs(self):
        with patch.dict(os.environ,{},clear=True),patch.object(bench.subprocess,'run',side_effect=AssertionError('A job/API process started')):
            code=self.invoke('evaluate','--runs-root',str(self.experiment),'--submit')
        self.assertEqual(code,2);self.assertFalse(self.experiment.exists())

    def test_pixel_only_evaluates_every_model_then_compares_without_judge(self):
        self.make_protocols()
        with patch.object(bench,'run_command',side_effect=self.local):
            code=self.invoke('evaluate','--runs-root',str(self.experiment),'--pixel-only')
        self.assertEqual(code,0)
        basenames=[Path(c[1]).name for c in self.calls]
        self.assertEqual(basenames,['evaluate.py','preview_run.py','formal_report.py']*2+['compare_models.py'])
        self.assertTrue(all('--pixel-only' in c for c in self.calls if Path(c[1]).name in ('formal_report.py','compare_models.py')))
        for model in self.models:self.assertTrue((self.experiment/model/'stages_all.json').exists())

    def test_dry_run_has_no_real_scoring_flag_and_compares_pixel_track(self):
        self.make_protocols()
        with patch.object(bench,'run_command',side_effect=self.local):
            code=self.invoke('evaluate','--runs-root',str(self.experiment),'--dry-run')
        self.assertEqual(code,0)
        judges=[c for c in self.calls if Path(c[1]).name=='mllm_evaluate.py']
        self.assertEqual(len(judges),2)
        for command in judges:
            self.assertIn('--dry-run',command);self.assertNotIn('--require-scoring',command)
            self.assertTrue(command[command.index('--output')+1].endswith('mllm_pending.json'))
        self.assertIn('--pixel-only',self.calls[-1])

    def test_real_evaluation_uses_config_and_requires_semantic_completion(self):
        self.make_protocols();config=self.root/'private judge.json';config.write_text('{}')
        with patch.object(bench,'judge_ready') as ready,patch.object(bench,'run_command',side_effect=self.local):
            code=self.invoke('evaluate','--runs-root',str(self.experiment),'--config',str(config))
        self.assertEqual(code,0);ready.assert_called_once_with(config.resolve())
        for command in self.calls:
            name=Path(command[1]).name
            if name=='mllm_evaluate.py':
                self.assertIn('--require-scoring',command);self.assertIn(str(config),command)
            if name=='formal_report.py':self.assertIn('--require-semantic',command)
        self.assertNotIn('--pixel-only',self.calls[-1])

    def test_one_model_stage_failure_does_not_skip_other_models_or_report(self):
        self.make_protocols()
        def fail_first(command):
            self.local(command)
            return 1 if Path(command[1]).name=='evaluate.py' and str(self.experiment/self.models[0]) in command[command.index('--generation-manifest')+1] else 0
        with patch.object(bench,'run_command',side_effect=fail_first):
            code=self.invoke('evaluate','--runs-root',str(self.experiment),'--pixel-only')
        self.assertEqual(code,1)
        self.assertEqual(sum(Path(c[1]).name=='formal_report.py' for c in self.calls),2)
        self.assertEqual(Path(self.calls[-1][1]).name,'compare_models.py')

    def test_generate_submit_freezes_one_script_per_model_and_reuses_active_jobs(self):
        with patch.object(bench.subprocess,'run',side_effect=self.scheduler):
            self.assertEqual(self.invoke(*self.generate_args('--submit','--gpu-constraint','h200')),0)
            first_calls=sum(c[0]=='sbatch' for c in self.calls)
            self.assertEqual(first_calls,2)
            self.assertEqual(self.invoke(*self.generate_args('--submit','--gpu-constraint','h200')),0)
            self.assertEqual(sum(c[0]=='sbatch' for c in self.calls),first_calls)
        jobs=json.loads((self.experiment/'jobs.json').read_text())
        self.assertEqual(set(jobs['entries']),{'generate:'+model for model in self.models})
        paths={entry['script'] for entry in jobs['entries'].values()};self.assertEqual(len(paths),2)
        for path in paths:
            text=Path(path).read_text()
            self.assertIn('#SBATCH --gres=gpu:1',text);self.assertIn('#SBATCH --constraint=h200',text)
            self.assertIn('#SBATCH --mem=256G',text);self.assertIn('#SBATCH --time=20:00:00',text)
            self.assertIn('--ids case000,case001',text)
            self.assertIn("'"+str(self.experiment),text)

    def test_default_local_generation_accepts_any_allocated_cuda_gpu(self):
        self.plan_local()
        self.assertTrue(self.calls)
        self.assertTrue(all('--require-h200' not in command for command in self.calls))

    def test_slurm_resource_flags_are_applied_and_hardware_changes_do_not_duplicate_jobs(self):
        arguments=self.generate_args('--submit','--partition','compute','--gpu-constraint','blackwell',
                                     '--cpus-per-task','4','--memory','128G')
        with patch.object(bench.subprocess,'run',side_effect=self.scheduler):
            self.assertEqual(self.invoke(*arguments),0)
            before=sum(c[0]=='sbatch' for c in self.calls)
            changed=list(arguments);changed[changed.index('blackwell')]='h200'
            self.assertEqual(self.invoke(*changed),2)
            self.assertEqual(sum(c[0]=='sbatch' for c in self.calls),before)
        jobs=json.loads((self.experiment/'jobs.json').read_text())
        for entry in jobs['entries'].values():
            text=Path(entry['script']).read_text()
            for expected in ('#SBATCH --partition=compute','#SBATCH --constraint=blackwell',
                             '#SBATCH --cpus-per-task=4','#SBATCH --mem=128G'):
                self.assertIn(expected,text)

    def test_cpu_offload_flag_reaches_each_generator(self):
        with patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke(*self.generate_args('--cpu-offload','--preflight')),0)
        self.assertTrue(all('--cpu-offload' in command for command in self.calls))

    def test_submit_chains_generation_gpu_pixels_cpu_judge_and_comparison(self):
        with patch.object(bench.subprocess,'run',side_effect=self.scheduler):
            self.assertEqual(self.invoke(*self.generate_args('--submit')),0)
            self.assertEqual(self.invoke('evaluate','--runs-root',str(self.experiment),'--dry-run','--submit'),0)
        jobs=json.loads((self.experiment/'jobs.json').read_text());entries=jobs['entries']
        self.assertEqual(len(entries),7)
        for model in self.models:
            generation=entries['generate:'+model];pixel=entries['pixel:'+model];judge=entries['judge:'+model]
            self.assertEqual(pixel['dependencies'],[generation['job_id']])
            self.assertEqual(judge['dependencies'],[pixel['job_id']])
            gpu_text=Path(pixel['script']).read_text();cpu_text=Path(judge['script']).read_text()
            self.assertIn('#SBATCH --gres=gpu:1',gpu_text);self.assertIn('--stage pixel',gpu_text)
            self.assertNotIn('#SBATCH --gres',cpu_text);self.assertIn('--stage judge',cpu_text)
            self.assertIn('CUDA_VISIBLE_DEVICES=""',cpu_text);self.assertIn('--dry-run',cpu_text)
        self.assertEqual(set(entries['comparison']['dependencies']),{entries['judge:'+m]['job_id'] for m in self.models})
        submits=[c for c in self.calls if c[0]=='sbatch']
        self.assertTrue(all(any(v.startswith('--dependency=afterany:') for v in c) for c in submits[2:]))

    def test_submit_config_path_is_quoted_and_key_never_enters_script(self):
        self.plan_local();config=self.root/'judge $(literal) config.json';config.write_text('{}')
        secret='fixture-private-secret-never-script'
        with patch.dict(os.environ,{'MLLM_API_KEY':secret,'MLLM_MODEL':'fixture-judge'}),\
                patch.object(bench.subprocess,'run',side_effect=self.scheduler):
            self.assertEqual(self.invoke('evaluate','--runs-root',str(self.experiment),'--config',str(config),'--submit'),0)
        for path in (self.experiment/'slurm').glob('*.slurm'):
            text=path.read_text();self.assertNotIn(secret,text)
            if '--stage judge' in text:self.assertIn("'"+str(config)+"'",text)

    def test_active_changed_submission_is_rejected(self):
        with patch.object(bench.subprocess,'run',side_effect=self.scheduler):
            self.assertEqual(self.invoke(*self.generate_args('--submit')),0)
            before=sum(c[0]=='sbatch' for c in self.calls)
            self.assertEqual(self.invoke(*self.generate_args('--submit','--time','10:00:00')),2)
        self.assertEqual(sum(c[0]=='sbatch' for c in self.calls),before)

    def test_completed_job_purged_from_squeue_does_not_block_resume(self):
        replies=[SimpleNamespace(returncode=1,stdout='',stderr='slurm_load_jobs error: Invalid job id specified'),
                 SimpleNamespace(returncode=0,stdout='COMPLETED\n',stderr='')]
        with patch.object(bench.subprocess,'run',side_effect=replies) as process:
            self.assertFalse(bench.active_job('10184'))
        self.assertEqual(process.call_args_list[1].args[0][0],'sacct')

    def test_purged_job_with_no_accounting_row_is_inactive(self):
        replies=[SimpleNamespace(returncode=1,stdout='',stderr='Invalid job id specified'),
                 SimpleNamespace(returncode=0,stdout='',stderr='')]
        with patch.object(bench.subprocess,'run',side_effect=replies):self.assertFalse(bench.active_job('100'))

    def test_scheduler_connection_failure_is_not_mistaken_for_finished_job(self):
        replies=[SimpleNamespace(returncode=1,stdout='',stderr='Unable to contact slurm controller'),
                 SimpleNamespace(returncode=1,stdout='',stderr='Accounting unavailable')]
        with patch.object(bench.subprocess,'run',side_effect=replies),self.assertRaisesRegex(ValueError,'refusing a duplicate'):
            bench.active_job('100')

    def test_explicit_compare_selects_requested_models_and_uses_shared_scope(self):
        self.make_protocols()
        with patch.object(bench,'run_command',side_effect=self.local):
            code=self.invoke('compare','--runs-root',str(self.experiment),'--pixel-only')
        self.assertEqual(code,0);command=self.calls[-1]
        self.assertEqual(command[command.index('--models')+1],','.join(self.models))
        self.assertEqual(command[command.index('--expected-ids')+1],'all')

    def test_download_is_an_explicit_subcommand_and_env_check_uses_profile(self):
        with patch.object(bench,'run_command',side_effect=self.local):
            self.assertEqual(self.invoke('download','--models',self.models[0],'--metadata-only'),0)
            self.assertEqual(self.invoke('check-env'),0)
        self.assertEqual(Path(self.calls[0][1]).name,'download_models.py')
        self.assertIn('--metadata-only',self.calls[0])
        program=self.calls[1][2]
        for model in bench.model_ids():self.assertIn(bench.MODEL_CONFIGS[model]['pipeline'],program)
        self.assertNotIn('LearnedPerceptualImagePatchSimilarity(net_type',program)

    def test_python_path_preserves_venv_symlink(self):
        target=self.root/'basepython';target.write_text('fixture')
        link=self.root/'venv/bin/python';link.parent.mkdir(parents=True);link.symlink_to(target)
        self.assertEqual(bench.python_path(link),str(link));self.assertNotEqual(bench.python_path(link),str(target))


if __name__=='__main__':unittest.main()
