"""Simple commands for local-model generation, common scoring and comparison."""
from __future__ import annotations

from settings import DATASET_ROOT, MODEL_ROOT

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

from evaluation_binding import canonical_hash,load_tasks,resolve_track_root
from generate_models import select_tasks
from model_registry import MODEL_CONFIGS
import model_registry

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=Path(__file__).resolve().parent
ACTIVE_STATES={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD'}


def now():return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


@contextmanager
def experiment_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.bench.lock').open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another command is updating this experiment; retry after it finishes planning') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def read_json(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def python_path(value):
    # Preserve a venv's executable symlink. resolve() would select its base
    # interpreter and lose the venv's installed packages.
    return os.path.abspath(os.path.expanduser(str(value or sys.executable)))


def model_ids(selector='all',available=None):
    defaults=list(getattr(model_registry,'DEFAULT_MODELS',tuple(MODEL_CONFIGS)))
    allowed=list(available) if available is not None else defaults
    selected=allowed if selector=='all' else selector.split(',')
    if not selected or any(not value or value!=value.strip() for value in selected) or len(set(selected))!=len(selected):
        raise ValueError('--models requires all or distinct comma-separated IDs')
    unknown=set(selected)-set(allowed if available is not None else MODEL_CONFIGS)
    if unknown:raise ValueError('Unknown model IDs: '+', '.join(sorted(unknown)))
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) for name in selected):
        raise ValueError('Invalid model ID')
    if available is not None and set(selected)-set(allowed):raise ValueError('Selected model is not in this experiment')
    return selected


def run_command(command):
    return subprocess.run([str(value) for value in command],check=False).returncode


def judge_ready(config):
    """Fail before scoring/submission; do not persist or display credentials."""
    from mllm_evaluate import parser,load_credentials
    arguments=['--predictions',str(ROOT),'--output',str(ROOT/'UNUSED')]
    if config is not None:arguments.extend(['--config',str(config)])
    credentials=load_credentials(parser().parse_args(arguments))
    if not credentials['api_key'] or not credentials['model']:
        raise ValueError('Formal semantic evaluation requires a judge API key and model. Set --config with api_key_env/api_key_file, or use --pixel-only/--dry-run.')
    if not 0<=credentials['max_retries']<=10 or not credentials['timeout']>0:
        raise ValueError('Invalid judge retry/timeout configuration')


def check_environment(args):
    pipelines=list(dict.fromkeys(MODEL_CONFIGS[name]['pipeline'] for name in model_ids()))
    program='''import importlib,json
modules=['numpy','PIL','cv2','torch','torchvision','torchmetrics','transformers','diffusers','accelerate','huggingface_hub','tabulate']
versions={}
for name in modules:
 module=importlib.import_module(name);versions[name]=getattr(module,'__version__','available')
'''
    program+='\nimport diffusers\npipelines='+repr(pipelines)+'\nfor name in pipelines:getattr(diffusers,name)\n'
    program+="report={'environment_ok':True,'versions':versions,'pipelines':pipelines,'model_weights_loaded':False}\n"
    if args.prepare_metrics:
        program+='''from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
metric=LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to('cpu')
report['lpips_weights_ready']=True
'''
    program+="print(json.dumps(report,indent=2))\n"
    return run_command([args.python,'-c',program])


def _safe_path(path):
    if '\n' in str(path) or '\r' in str(path):raise ValueError('Paths cannot contain line breaks')
    return Path(path).resolve()


def plan_experiment(args):
    root=_safe_path(args.output_root);dataset=resolve_track_root(args.dataset_root,getattr(args,'track',None))
    if args.smoke or getattr(args,'sample',None) is not None:
        from sampling import sample_tasks
        count=7 if args.smoke else args.sample
        tasks=sample_tasks(select_tasks(dataset,None,True),count,args.sample_seed)
    else:tasks=select_tasks(dataset,args.ids,args.all)
    selected=model_ids(args.models)
    experiment={'schema_version':1,'dataset_root':str(dataset),'case_ids':[task['id'] for task in tasks],
                'run_id':args.run_id or root.name,'models':selected,'resolution_protocol':'native-pad32-v1'}
    if args.smoke or getattr(args,'sample',None) is not None:
        experiment['sampling']={'method':'operation-stratified-source-diverse-v1','count':count,'seed':args.sample_seed}
    if not experiment['run_id'].strip():raise ValueError('Run ID cannot be empty')
    path=root/'experiment.json';previous=read_json(path)
    if previous is not None:
        for key in ('schema_version','dataset_root','case_ids','run_id','resolution_protocol'):
            if previous.get(key)!=experiment[key]:raise ValueError('Experiment dataset/selection/run identity changed; use a new output root')
        experiment['models']=list(dict.fromkeys([*previous.get('models',[]),*selected]))
    write_json(path,experiment)
    return root,dataset,experiment,selected


def generate_command(args,root,dataset,experiment,model):
    command=[args.python,str(SCRIPTS/'generate_models.py'),'--model',model,'--dataset-root',str(dataset),
             '--output-dir',str(root/model),'--run-id',experiment['run_id'],
        '--ids',','.join(experiment['case_ids'])]
    if getattr(args,'cpu_offload',None) is not None:
        command.append('--cpu-offload' if args.cpu_offload else '--no-cpu-offload')
    if args.preflight:command.append('--preflight')
    if getattr(args,'model_config',None) is not None:command.extend(['--model-config',str(args.model_config)])
    return command


def active_job(job_id):
    if not str(job_id).isdigit():raise ValueError('Invalid recorded Slurm job ID')
    result=subprocess.run(['squeue','-h','-j',str(job_id),'-o','%T'],check=False,capture_output=True,text=True)
    if result.returncode!=0:
        # slurmctld drops completed jobs after MinJobAge; squeue then returns
        # "Invalid job id specified". Historical jobs must not block a later
        # judge run or a legitimate resume of this experiment.
        history=subprocess.run(['sacct','-n','-X','-j',str(job_id),'--format=State','--parsable2'],
                               check=False,capture_output=True,text=True)
        states=[line.split('|',1)[0].strip().split()[0].rstrip('+')
                for line in history.stdout.splitlines() if line.strip()]
        if history.returncode==0 and states:
            if any(state in ACTIVE_STATES for state in states):return True
            terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
                      'BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED','SPECIAL_EXIT'}
            if all(state in terminal for state in states):return False
        if 'invalid job id' in result.stderr.lower():return False
        raise ValueError('Could not check Slurm job state; refusing a duplicate submission')
    return any(line.strip() in ACTIVE_STATES for line in result.stdout.splitlines())


def _job_script(root,label,command,*,gpu,cpus,memory,time_limit,partition='gpu',gpu_constraint=None):
    if not re.fullmatch(r'(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]',time_limit):
        raise ValueError('Time must be HH:MM:SS or D-HH:MM:SS')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',partition):raise ValueError('Invalid Slurm partition')
    if gpu_constraint is not None and not re.fullmatch(r'[A-Za-z0-9_.-]+',gpu_constraint):
        raise ValueError('Invalid Slurm GPU constraint')
    if not isinstance(cpus,int) or cpus<1:raise ValueError('Slurm CPU count must be positive')
    if not re.fullmatch(r'[0-9]+[KMGTP]?',memory,re.IGNORECASE):raise ValueError('Invalid Slurm memory')
    folder=root/'slurm';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(label+'-'+uuid.uuid4().hex[:12]+'.slurm')
    log=root/'logs';log.mkdir(parents=True,exist_ok=True)
    lines=['#!/bin/bash','#SBATCH --job-name='+label,'#SBATCH --partition='+partition,
           '#SBATCH --cpus-per-task='+str(cpus),'#SBATCH --mem='+memory,'#SBATCH --time='+time_limit,
           '#SBATCH --output='+shlex.quote(str(log/(label+'-%j.out'))),
           '#SBATCH --error='+shlex.quote(str(log/(label+'-%j.err')))]
    lines.append('#SBATCH --gres=gpu:'+('1' if gpu else '0'))
    if gpu_constraint:lines.append('#SBATCH --constraint='+gpu_constraint)
    lines.extend(['set -euo pipefail','cd '+shlex.quote(str(ROOT)),
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS='+str(cpus)+' MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1'])
    if not gpu:lines.append('export CUDA_VISIBLE_DEVICES=""')
    lines.append(shlex.join([str(value) for value in command]));path.write_text('\n'.join(lines)+'\n')
    return path


def submit_job(root,jobs,key,label,command,*,gpu,cpus,memory,time_limit,dependencies=None,partition='gpu',gpu_constraint=None):
    request={'command':[str(v) for v in command],'gpu':gpu,'cpus':cpus,'memory':memory,'time':time_limit}
    request.update(partition=partition,gpu_constraint=gpu_constraint)
    fingerprint=canonical_hash(request);previous=jobs['entries'].get(key)
    if previous and active_job(previous['job_id']):
        if previous.get('request_sha256')!=fingerprint:raise ValueError('An active Slurm job already has another configuration: '+key)
        return str(previous['job_id'])
    script=_job_script(root,label,command,gpu=gpu,cpus=cpus,memory=memory,time_limit=time_limit,
                       partition=partition,gpu_constraint=gpu_constraint)
    command_line=['sbatch','--parsable']
    dependencies=[str(job) for job in (dependencies or []) if job]
    if dependencies:command_line.append('--dependency=afterany:'+':'.join(dict.fromkeys(dependencies)))
    command_line.append(str(script))
    result=subprocess.run(command_line,check=False,capture_output=True,text=True)
    if result.returncode!=0:raise ValueError('Slurm submission failed: '+result.stderr.strip())
    job_id=result.stdout.strip().split(';',1)[0]
    if not job_id.isdigit():raise ValueError('Slurm returned an invalid job ID')
    entry={'job_id':job_id,'script':str(script),'submitted_at':now(),'request_sha256':fingerprint,
           'dependencies':dependencies,**request}
    jobs['entries'][key]=entry;jobs.setdefault('history',[]).append({'key':key,**entry})
    write_json(root/'jobs.json',jobs)
    print(json.dumps({'stage':key,'job_id':job_id,'script':str(script)}),flush=True)
    return job_id


def load_jobs(root):
    value=read_json(root/'jobs.json',{'schema_version':1,'entries':{},'history':[]})
    if value.get('schema_version')!=1 or not isinstance(value.get('entries'),dict):raise ValueError('Invalid experiment jobs.json')
    return value


def generate(args):
    root=_safe_path(args.output_root)
    with experiment_lock(root):
        root,dataset,experiment,selected=plan_experiment(args)
        commands={model:generate_command(args,root,dataset,experiment,model) for model in selected}
        if args.submit:
            if args.preflight:raise ValueError('--preflight is a CPU check; omit --submit')
            jobs=load_jobs(root)
            for model in selected:submit_job(root,jobs,'generate:'+model,'gen-'+model,commands[model],
                gpu=True,cpus=args.cpus_per_task,memory=args.memory,time_limit=args.time,
                partition=args.partition,gpu_constraint=args.gpu_constraint)
            return 0
    results={}
    for model in selected:
        print(json.dumps({'stage':'generate','model':model}),flush=True)
        results[model]=run_command(commands[model])
    write_json(root/('preflight_results.json' if args.preflight else 'generation_stages.json'),
               {'mode':'preflight' if args.preflight else 'generation','models':results,'updated_at':now()})
    return 0 if all(code==0 for code in results.values()) else 1


def evaluation_context(args):
    root=_safe_path(args.runs_root);experiment=read_json(root/'experiment.json')
    single=read_json(root/'protocol.json')
    if experiment:
        dataset=resolve_track_root(args.dataset_root or experiment['dataset_root'],getattr(args,'track',None))
        if str(dataset)!=experiment['dataset_root']:raise ValueError('Evaluation dataset differs from experiment')
        available=experiment['models'];expected=experiment['case_ids']
    elif single:
        dataset=resolve_track_root(args.dataset_root or DATASET_ROOT,getattr(args,'track',None));available=[single['model_id']];expected=single['case_ids']
    else:
        dataset=resolve_track_root(args.dataset_root or DATASET_ROOT,getattr(args,'track',None))
        protocols=[read_json(path) for path in root.glob('*/protocol.json')]
        available=[p['model_id'] for p in protocols if p]
        if not available:raise ValueError('No experiment.json or generated model runs found')
        expected=protocols[0]['case_ids']
    selected=model_ids(args.models,available)
    tasks=load_tasks(dataset)
    if not expected or len(expected)!=len(set(expected)) or set(expected)-set(tasks):raise ValueError('Invalid experiment case selection')
    runs={model:root if single else root/model for model in selected}
    for model,run in runs.items():
        protocol=read_json(run/'protocol.json')
        if protocol and (protocol.get('model_id')!=model or set(protocol.get('case_ids',[]))!=set(expected)):
            raise ValueError('Model run identity/case set differs from experiment: '+model)
    expected_option='all' if set(expected)==set(tasks) else ','.join(expected)
    return root,dataset,selected,runs,expected_option


def evaluation_commands(args,run,dataset,expected):
    common=[args.python]
    pixel=common+[str(SCRIPTS/'evaluate.py'),'--dataset-root',str(dataset),'--generation-manifest',str(run/'generation_manifest.json'),
                  '--ids',','.join(load_tasks(dataset)) if expected=='all' else expected,'--device',args.device,
                  '--cpu-threads','8' if args.device=='cuda' else '2','--output',str(run/'pixel_scores.json')]
    preview=common+[str(SCRIPTS/'preview_run.py'),'--dataset-root',str(dataset),'--generation-manifest',str(run/'generation_manifest.json'),
                    '--output',str(run/'comparison.png')]
    judge=common+[str(SCRIPTS/'mllm_evaluate.py'),'--dataset',str(dataset),'--generation-manifest',str(run/'generation_manifest.json'),
                  '--ids',','.join(load_tasks(dataset)) if expected=='all' else expected,
                  '--output',str(run/('mllm_pending.json' if args.dry_run else 'mllm_scores.json'))]
    if args.config is not None:judge.extend(['--config',str(args.config)])
    if not getattr(args,'resume',True):
        pixel.append('--no-resume');judge.append('--no-resume')
    judge.extend(['--workers',str(getattr(args,'workers',5)),'--text-profile',getattr(args,'text_profile','strict')])
    if not getattr(args,'progress',True):judge.append('--no-progress')
    judge.append('--dry-run' if args.dry_run else '--require-scoring')
    formal=common+[str(SCRIPTS/'formal_report.py'),'--run-dir',str(run),'--dataset-root',str(dataset),'--expected-ids',expected]
    if args.pixel_only:formal.append('--pixel-only')
    elif args.dry_run:formal.append('--dry-run')
    else:formal.append('--require-semantic')
    return {'pixel':pixel,'preview':preview,'judge':judge,'formal':formal}


def _local_evaluate(args,dataset,selected,runs,expected):
    results={}
    for model in selected:
        run=runs[model];run.mkdir(parents=True,exist_ok=True)
        commands=evaluation_commands(args,run,dataset,expected);stages={}
        with (run/'.evaluation.lock').open('a+b') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                # Protect the active writer and let other independent models run.
                results[model]={'busy':1}
                print(json.dumps({'model':model,'status':'busy',
                    'error':'Another evaluation is writing this model run; its files are unchanged'}),flush=True)
                continue
            try:
                if args.stage in ('all','pixel'):
                    for name in ('pixel','preview'):stages[name]=run_command(commands[name])
                if args.stage in ('all','judge'):
                    if not args.pixel_only:stages['judge']=run_command(commands['judge'])
                    stages['formal']=run_command(commands['formal'])
                write_json(run/('stages_'+args.stage+'.json'),{'model_id':model,'stages':stages,'updated_at':now()})
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
        results[model]=stages
    return results


def compare_command(args,root,dataset,selected,expected=None):
    command=[args.python,str(SCRIPTS/'compare_models.py'),'--runs-root',str(root),'--dataset-root',str(dataset),
             '--models',','.join(selected),'--output-dir',str(args.output_dir or root/'reports')]
    if args.pixel_only or getattr(args,'dry_run',False):command.append('--pixel-only')
    if expected:command.extend(['--expected-ids',expected])
    return command


def _stage_command(args,root,dataset,model,stage):
    command=[args.python,str(SCRIPTS/'bench.py'),'evaluate','--runs-root',str(root),'--dataset-root',str(dataset),
             '--models',model,'--stage',stage,'--no-compare','--device',args.device if stage=='pixel' else 'cpu']
    if args.config is not None:command.extend(['--config',str(args.config)])
    if not getattr(args,'resume',True):command.append('--no-resume')
    command.extend(['--workers',str(getattr(args,'workers',5)),
                    '--text-profile',getattr(args,'text_profile','strict')])
    if not getattr(args,'progress',True):command.append('--no-progress')
    if args.pixel_only:command.append('--pixel-only')
    elif args.dry_run:command.append('--dry-run')
    return command


def evaluate_predictions(args):
    """Score native images from any model through the same two evaluators."""
    if args.submit or args.stage!='all':raise ValueError('Use local complete evaluation for --predictions')
    if args.models!='all':raise ValueError('--predictions evaluates one model folder per command; omit --models')
    predictions=_safe_path(args.predictions)
    if not predictions.is_dir():raise ValueError('Predictions directory does not exist')
    dataset=resolve_track_root(args.dataset_root or DATASET_ROOT,getattr(args,'track',None))
    tasks=select_tasks(dataset,args.ids,args.ids is None)
    ids=','.join(task['id'] for task in tasks)
    output=_safe_path(args.output_dir or ROOT/'results'/predictions.name)
    if output==dataset or output.is_relative_to(dataset):raise ValueError('Results must be outside the dataset directory')
    pixel=[args.python,str(SCRIPTS/'evaluate.py'),'--dataset-root',str(dataset),
           '--predictions',str(predictions),'--ids',ids,'--device',args.device,
           '--cpu-threads','8' if args.device=='cuda' else '2','--output',str(output/'pixel_scores.json')]
    judge=[args.python,str(SCRIPTS/'mllm_evaluate.py'),'--dataset',str(dataset),
           '--predictions',str(predictions),'--ids',ids,
           '--output',str(output/('mllm_pending.json' if args.dry_run else 'mllm_scores.json'))]
    if args.config is not None:judge.extend(['--config',str(args.config)])
    if not getattr(args,'resume',True):
        pixel.append('--no-resume');judge.append('--no-resume')
    judge.extend(['--workers',str(getattr(args,'workers',5)),'--text-profile',getattr(args,'text_profile','strict')])
    if not getattr(args,'progress',True):judge.append('--no-progress')
    judge.append('--dry-run' if args.dry_run else '--require-scoring')
    with experiment_lock(output):
        stages={'pixel':run_command(pixel)}
        if not args.pixel_only:stages['judge']=run_command(judge)
    failed=any(stages.values())
    print(json.dumps({'status':'failed' if failed else 'validated' if args.dry_run else 'complete','output_dir':str(output),
                      'mode':'pixel-only' if args.pixel_only else 'dry-run' if args.dry_run else 'formal',
                      'cases':len(tasks),'stages':stages}),flush=True)
    return 1 if failed else 0


def evaluate(args):
    if not 1<=args.workers<=32:raise ValueError('--workers must be in [1,32]')
    if not args.pixel_only and not args.dry_run and args.stage!='pixel':judge_ready(args.config)
    if args.predictions is not None:return evaluate_predictions(args)
    if args.ids is not None:raise ValueError('--ids is for --predictions; generated runs already record their case selection')
    root,dataset,selected,runs,expected=evaluation_context(args)
    if args.submit:
        if args.stage!='all':raise ValueError('Submit the complete evaluation stages together')
        with experiment_lock(root):
            jobs=load_jobs(root);completed=[]
            for model in selected:
                prior=jobs['entries'].get('generate:'+model)
                dependency=[prior['job_id']] if prior and active_job(prior['job_id']) else []
                pixel=submit_job(root,jobs,'pixel:'+model,'pixels-'+model,_stage_command(args,root,dataset,model,'pixel'),
                    gpu=args.device=='cuda',cpus=args.cpus_per_task,memory=args.memory,time_limit='02:00:00',dependencies=dependency,
                    partition=args.partition,gpu_constraint=args.gpu_constraint)
                judge=submit_job(root,jobs,'judge:'+model,'judge-'+model,_stage_command(args,root,dataset,model,'judge'),
                    gpu=False,cpus=2,memory='16G',time_limit=args.time,dependencies=[pixel],partition=args.partition,
                    gpu_constraint=args.gpu_constraint)
                completed.append(judge)
            command=compare_command(args,root,dataset,selected,expected)
            submit_job(root,jobs,'comparison','compare-edit-models',command,gpu=False,cpus=2,memory='16G',
                       time_limit='02:00:00',dependencies=completed,partition=args.partition,
                       gpu_constraint=args.gpu_constraint)
        return 0
    results=_local_evaluate(args,dataset,selected,runs,expected)
    failed=any(code for stages in results.values() for code in stages.values())
    if not args.no_compare and args.stage=='all':
        failed=bool(run_command(compare_command(args,root,dataset,selected,expected))) or failed
    print(json.dumps({'mode':'pixel-only' if args.pixel_only else 'dry-run' if args.dry_run else 'formal',
                      'model_stages':results,'semantic_scoring_requested':not args.pixel_only and not args.dry_run}),flush=True)
    return 1 if failed else 0


def compare(args):
    root,dataset,selected,_,expected=evaluation_context(args)
    return run_command(compare_command(args,root,dataset,selected,expected))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    listing=sub.add_parser('list-models',help='List built-in and additional Diffusers model profiles')
    listing.add_argument('--model-config',type=Path)
    listing.add_argument('--python',default=sys.executable)
    check=sub.add_parser('check-env',help='Check installed dependencies and pipeline imports')
    check.add_argument('--python',default=sys.executable);check.add_argument('--prepare-metrics',action='store_true')
    download=sub.add_parser('download',help='Download pinned registered checkpoints')
    download.add_argument('--models',default='all');download.add_argument('--output-root',type=Path,default=MODEL_ROOT)
    download.add_argument('--model-config',type=Path,help='Additional Diffusers model profiles in JSON')
    download.add_argument('--metadata-only',action='store_true');download.add_argument('--max-workers',type=int,default=4)
    download.add_argument('--python',default=sys.executable)
    generate_p=sub.add_parser('generate',help='Generate each selected model into an isolated directory')
    generate_p.add_argument('--models',default='all');generate_p.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    generate_p.add_argument('--track',choices=('text','asset'),help='Dataset track; text by default, asset selects the assets subdirectory')
    generate_p.add_argument('--model-config',type=Path,help='Additional Diffusers model profiles in JSON')
    generate_p.add_argument('--output-root',required=True,type=Path);generate_p.add_argument('--run-id')
    scope=generate_p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--all',action='store_true');scope.add_argument('--ids');scope.add_argument('--smoke',action='store_true')
    scope.add_argument('--sample',type=int,help='Deterministic operation-stratified subset, prioritizing distinct source designs')
    generate_p.add_argument('--sample-seed',type=int,default=42,help='Subset selection seed; does not change model inference seed')
    generate_p.add_argument('--preflight',action='store_true');generate_p.add_argument('--submit',action='store_true')
    generate_p.add_argument('--python',default=sys.executable);generate_p.add_argument('--time',default='20:00:00')
    generate_p.add_argument('--cpu-offload',action=argparse.BooleanOptionalAction,default=None)
    generate_p.add_argument('--partition',default='gpu');generate_p.add_argument('--gpu-constraint')
    generate_p.add_argument('--cpus-per-task',type=int,default=8);generate_p.add_argument('--memory',default='256G')
    eval_p=sub.add_parser('evaluate',help='Compute common pixels, judge dimensions, previews and formal reports')
    inputs=eval_p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs-root',type=Path,help='Experiment directory produced by generate')
    inputs.add_argument('--predictions',type=Path,help='Native RGB <case-id>.png images from any model')
    eval_p.add_argument('--ids',help='Comma-separated subset for --predictions; default all dataset cases')
    eval_p.add_argument('--dataset-root',type=Path)
    eval_p.add_argument('--track',choices=('text','asset'),help='Select a track for external predictions; generated runs remember their dataset')
    eval_p.add_argument('--models',default='all');eval_p.add_argument('--config',type=Path)
    eval_p.add_argument('--text-profile',choices=('strict','texteditbench'),default='strict',help='Text rubric; strict by default, texteditbench reproduces the original rubric')
    eval_p.add_argument('--no-resume',action='store_false',dest='resume',help='Force recomputation; default skips validated successful pixel metrics and judge dimensions')
    eval_p.add_argument('--workers',type=int,default=5,help='Concurrent judge cases / maximum simultaneous API requests (1-32; default 5)')
    eval_p.add_argument('--progress',action=argparse.BooleanOptionalAction,default=True,help='Show judge progress (default enabled)')
    modes=eval_p.add_mutually_exclusive_group();modes.add_argument('--pixel-only',action='store_true');modes.add_argument('--dry-run',action='store_true')
    eval_p.add_argument('--device',choices=('cpu','cuda'),default='cpu');eval_p.add_argument('--submit',action='store_true')
    eval_p.add_argument('--partition',default='gpu');eval_p.add_argument('--gpu-constraint')
    eval_p.add_argument('--cpus-per-task',type=int,default=8);eval_p.add_argument('--memory',default='64G')
    eval_p.add_argument('--time',default='20:00:00',help='Slurm judge-stage time limit')
    eval_p.add_argument('--python',default=sys.executable);eval_p.add_argument('--output-dir',type=Path)
    eval_p.add_argument('--stage',choices=('all','pixel','judge'),default='all',help=argparse.SUPPRESS)
    eval_p.add_argument('--no-compare',action='store_true',help=argparse.SUPPRESS)
    comp=sub.add_parser('compare',help='Revalidate runs and export comparable JSON/CSV scores')
    comp.add_argument('--runs-root',required=True,type=Path);comp.add_argument('--dataset-root',type=Path)
    comp.add_argument('--track',choices=('text','asset'))
    comp.add_argument('--models',default='all');comp.add_argument('--output-dir',type=Path)
    comp.add_argument('--pixel-only',action='store_true');comp.add_argument('--python',default=sys.executable)
    return p


def main(argv=None):
    args=parser().parse_args(argv);args.python=python_path(args.python)
    if getattr(args,'config',None) is not None:args.config=args.config.resolve()
    try:
        if getattr(args,'model_config',None) is not None:
            args.model_config=args.model_config.resolve()
            model_registry.load_model_config(args.model_config)
        if args.command=='list-models':
            print(json.dumps({'default_models':list(model_registry.DEFAULT_MODELS),'models':[
                {'id':name,'repo_id':config['repo_id'],'pipeline':config['pipeline'],
                 'steps':config['num_inference_steps']} for name,config in MODEL_CONFIGS.items()]},indent=2))
            return 0
        if args.command=='check-env':return check_environment(args)
        if args.command=='download':
            model_ids(args.models)
            command=[args.python,str(SCRIPTS/'download_models.py'),'--models',args.models,'--output-root',str(args.output_root.resolve()),
                     '--max-workers',str(args.max_workers)]
            if args.metadata_only:command.append('--metadata-only')
            if args.model_config is not None:command.extend(['--model-config',str(args.model_config)])
            return run_command(command)
        return {'generate':generate,'evaluate':evaluate,'compare':compare}[args.command](args)
    except Exception as exc:
        # Never echo credentials/config contents through arbitrary exceptions.
        reason=str(exc) if isinstance(exc,(ValueError,FileNotFoundError)) else type(exc).__name__
        print(json.dumps({'status':'error','error':reason}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
