"""Generate one registered editing model over selected independent cases.

This stage has no judge API. --preflight hashes/checks local model contents but
does not import torch or load weights into CPU/GPU memory.
"""
from settings import DATASET_ROOT, MODEL_ROOT
from contextlib import contextmanager
from datetime import datetime,timezone
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from evaluation_binding import canonical_hash
from generation_provenance import collect_generation_content
from model_registry import (
    MODEL_CONFIGS,GENERATION_PROTOCOL_VERSION,RESOLUTION_PROTOCOL,get_model_config,
    inference_kwargs,load_pipeline,validate_model_files,validate_pipeline_signature,extract_generation_output,load_model_config,
)
from output_adapter import instruction_sha256,load_prediction,prepare_rgb_record,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name)
            stream.write((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
            stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)


@contextmanager
def generation_lock(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.generation.lock').open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('This generation output already has an active writer; use a separate run directory') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def select_tasks(dataset,ids=None,all_cases=False):
    tasks=[json.loads(l) for l in (Path(dataset)/'tasks.jsonl').read_text().splitlines() if l.strip()]
    known={}
    for task in tasks:
        if not isinstance(task.get('id'),str) or not task['id'] or task['id'] in known:
            raise ValueError('Invalid or duplicate task ID')
        if not isinstance(task.get('image'),str) or not isinstance(task.get('instruction'),str) or not task['instruction'].strip():
            raise ValueError('Tasks require source image and nonempty instruction')
        known[task['id']]=task
    if not tasks:raise ValueError('Dataset has no tasks')
    if all_cases:return tasks
    requested=ids.split(',') if ids is not None else []
    if not requested or any(not x or x!=x.strip() for x in requested) or len(set(requested))!=len(requested):
        raise ValueError('--ids requires distinct comma-separated nonempty IDs')
    missing=[x for x in requested if x not in known]
    if missing:raise ValueError('Unknown case IDs: '+', '.join(missing))
    return [known[cid] for cid in requested]


def input_plan(dataset,tasks):
    rows=[]
    for task in tasks:
        source=(Path(dataset)/task['image']).resolve()
        with Image.open(source) as image:
            if image.mode!='RGB':raise ValueError(f'{task["id"]}: source must be RGB')
            w,h=image.size
        pw,ph=math.ceil(w/32)*32,math.ceil(h/32)*32
        rows.append({'id':task['id'],'source_file':str(source),'source_size':[w,h],
            'canvas_size':[pw,ph],'evaluation_roi':[0,0,w,h],
            'padding':{'left':0,'top':0,'right':pw-w,'bottom':ph-h,'mode':'edge' if (pw,ph)!=(w,h) else 'none'},
            'source_sha256':sha256_file(source),'instruction_sha256':instruction_sha256(task),
            'instruction':task['instruction']})
    return rows


def require_directory(out,protocol=None):
    """Return prior records only when model/code/selection identities agree."""
    path=out/'protocol.json';manifest=out/'generation_manifest.json';previous=None
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('generation_protocol_version')!=GENERATION_PROTOCOL_VERSION:
            raise ValueError('Existing directory belongs to another or legacy generation protocol; use a new run directory')
        fingerprint=previous.get('protocol_sha256')
        if canonical_hash({k:v for k,v in previous.items() if k!='protocol_sha256'})!=fingerprint:
            raise ValueError('Existing generation protocol body/fingerprint mismatch')
    elif manifest.exists():raise ValueError('Generation manifest exists without a pinned protocol')
    elif out.exists() and any(p.name!='.generation.lock' for p in out.iterdir()):
        raise ValueError('Nonempty output directory has no compatible generation protocol')
    if protocol is None:return {}
    if previous is not None and previous['protocol_sha256']!=protocol['protocol_sha256']:
        raise ValueError('Generation model/code/input/selection fingerprint changed; use a new run directory')
    if not manifest.exists():return {}
    saved=json.loads(manifest.read_text());content_sha=protocol['content_provenance']['sha256']
    for name,value in [('schema_version',1),('model_id',protocol['model_id']),('run_id',protocol['run_id']),
                       ('generation_protocol_version',GENERATION_PROTOCOL_VERSION),
                       ('protocol_sha256',protocol['protocol_sha256']),
                       ('content_provenance_sha256',content_sha),('expected_ids',protocol['case_ids'])]:
        if saved.get(name)!=value:raise ValueError('Generation manifest identity mismatch: '+name)
    records={}
    if not isinstance(saved.get('records'),list):raise ValueError('Generation records list missing')
    for record in saved['records']:
        cid=record.get('id')
        if cid not in protocol['case_ids'] or cid in records:raise ValueError('Unknown or duplicate generation record ID')
        for name in ('model_id','run_id','protocol_sha256'):
            if record.get(name)!=protocol[name]:raise ValueError('Generation record identity mismatch: '+name)
        if record.get('content_provenance_sha256')!=content_sha:raise ValueError('Generation record content mismatch')
        records[cid]=record
    return records


def build_protocol(args,config,tasks,plan,content):
    libraries={}
    for name in ('torch','diffusers','transformers','Pillow','numpy','accelerate'):
        try:libraries[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:libraries[name]='not_installed'
    folder=Path(__file__).resolve().parent
    sources={name:sha256_file(folder/name) for name in
             ('generate_models.py','model_registry.py','output_adapter.py','generation_provenance.py','flux3_adapter.py','settings.py')}
    value={'schema_version':2,'generation_protocol_version':GENERATION_PROTOCOL_VERSION,
        'model_id':args.model,'run_id':args.run_id or args.output_dir.name,'repo_id':config['repo_id'],
        'model_path':str(args.model_path),'pipeline':config['pipeline'],'model_config':config,
        'case_ids':[task['id'] for task in tasks],'task_inputs_sha256':canonical_hash(plan),
        'resolution_protocol':RESOLUTION_PROTOCOL,'content_provenance':content,
        'steps':config['num_inference_steps'],'sampling':config['call_kwargs'],'seed':args.seed,
        'dtype':'bfloat16','device':args.device,'cpu_offload':args.cpu_offload,
        'input':'Exactly one source image and its original instruction per independent call; no mask, reference or cross-case history.',
        'padding':'Right/bottom edge padding to32; native ROI only during evaluation. No input/output stretching.',
        'raw_policy':'Preserve returned PIL bytes and full canvas; separate exact RGB channels; no compositing/cropping/resampling.',
        'conditioning_policy':config['conditioning_policy'],'libraries':libraries,'runner_sources':sources}
    return {**value,'protocol_sha256':canonical_hash(value)}


def run_generation(args,config,protocol,tasks,plan,records,torch,*,pipeline_factory=load_pipeline):
    out=args.output_dir;dataset=args.dataset_root;manifest=out/'generation_manifest.json'
    fingerprint=protocol['protocol_sha256'];content_sha=protocol['content_provenance']['sha256']
    gpu=torch.cuda.get_device_name(args.device)
    def save(status,current=None,error=None):
        ordered=[records[t['id']] for t in tasks if t['id'] in records]
        write(manifest,{'schema_version':1,'model':str(args.model_path),'model_id':args.model,
            'run_id':protocol['run_id'],'generation_protocol_version':GENERATION_PROTOCOL_VERSION,
            'protocol_sha256':fingerprint,'content_provenance_sha256':content_sha,
            'expected_ids':protocol['case_ids'],'status':status,'updated_at':now(),
            'device':gpu,'records':ordered})
        write(out/'progress.json',{'model_id':args.model,'run_id':protocol['run_id'],'status':status,
            'current':current,'expected':len(tasks),'complete':sum(r['status']=='complete' for r in ordered),
            'failed':sum(r['status']!='complete' for r in ordered),'pending':len(tasks)-len(ordered),
            'error':error,'updated_at':now()})
    valid=set();invalid={}
    for task in tasks:
        previous=records.get(task['id'])
        if previous and previous.get('status')=='complete':
            try:load_prediction(task,dataset,record=previous,manifest_path=manifest)
            except Exception as exc:
                invalid[task['id']]=str(exc)
                if not args.repair_invalid:
                    records[task['id']]=dict(previous,status='invalid_output',error=str(exc))
            else:valid.add(task['id'])
    pending=[task for task in tasks if task['id'] not in valid and (args.repair_invalid or task['id'] not in invalid)]
    if not pending:
        failed=sum(records[t['id']]['status']!='complete' for t in tasks)
        save('failed' if failed else 'complete')
        return 1 if failed else 0
    save('loading_model')
    try:
        pipe=pipeline_factory(config,args.model_path,torch,args.device,args.cpu_offload)
        vae_factor=int(pipe.vae_scale_factor)
        if vae_factor<=0 or 32%(2*vae_factor):
            raise ValueError(f'Checkpoint VAE requires an output multiple incompatible with pad32: {2*vae_factor}')
    except Exception as exc:
        save('failed',error=f'Model loading failed: {type(exc).__name__}: {exc}');raise
    for task,p in zip(tasks,plan):
        cid=task['id']
        if cid in valid:continue
        if cid in invalid and not args.repair_invalid:continue
        attempt=1
        while (out/'outputs'/cid/f'attempt_{attempt:03}').exists():attempt+=1
        directory=out/'outputs'/cid/f'attempt_{attempt:03}';directory.mkdir(parents=True)
        metadata={k:v for k,v in p.items() if k not in ('source_file','instruction')}
        metadata.update(model_id=args.model,run_id=protocol['run_id'],model=str(args.model_path),
            pipeline=config['pipeline'],steps=config['num_inference_steps'],seed=args.seed,
            vae_scale_factor=vae_factor,packed_latent_multiple=2*vae_factor,
            attempt=attempt,protocol_sha256=fingerprint,content_provenance_sha256=content_sha,
            resolution_protocol=RESOLUTION_PROTOCOL,started_at=now(),device=gpu)
        if cid in invalid:metadata['repair_reason']=invalid[cid]
        save('generating',cid);start=time.monotonic()
        print(json.dumps({'id':cid,'model':args.model,'status':'generating','attempt':attempt}),flush=True)
        try:
            if sha256_file(p['source_file'])!=p['source_sha256']:raise ValueError('Dataset source changed after preflight')
            inputs=directory/'inputs';inputs.mkdir()
            copied=inputs/'source.png';shutil.copy2(p['source_file'],copied)
            if sha256_file(copied)!=p['source_sha256']:raise ValueError('Source copy differs from preflight bytes')
            write(inputs/'task.json',{'id':cid,'image':'source.png','instruction':task['instruction']})
            with Image.open(copied) as image:source=np.asarray(image).copy()
            w,h=p['source_size'];pw,ph=p['canvas_size']
            padded=Image.fromarray(np.pad(source,((0,ph-h),(0,pw-w),(0,0)),mode='edge'))
            padded_path=inputs/'model_input_padded.png';padded.save(padded_path)
            metadata.update(model_input=str(padded_path),model_input_sha256=sha256_file(padded_path))
            generator=torch.Generator(device=args.device).manual_seed(args.seed)
            kwargs=inference_kwargs(config,instruction=task['instruction'],image=padded,
                                    width=pw,height=ph,generator=generator)
            with torch.inference_mode():result=pipe(**kwargs)
            returned,prompt_metadata=extract_generation_output(config,result,task['instruction'])
            metadata.update(prompt_metadata)
            raw=directory/'model_output_original.png';returned.save(raw)
            record=prepare_rgb_record(raw,directory/'prediction.png',metadata)
        except Exception as exc:
            record=dict(metadata,status='error',error=f'{type(exc).__name__}: {exc}')
            raw=directory/'model_output_original.png'
            if raw.exists():record.update(raw_output=str(raw),raw_sha256=sha256_file(raw))
            torch.cuda.empty_cache()
        record.update(elapsed_seconds=round(time.monotonic()-start,3),finished_at=now())
        write(directory/'result.json',record);records[cid]=record;save('generating',cid)
        print(json.dumps({'id':cid,'status':record['status'],'error':record.get('error')}),flush=True)
    failed=sum(records[task['id']]['status']!='complete' for task in tasks)
    save('failed' if failed else 'complete')
    return 1 if failed else 0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='Built-in or custom registered model ID')
    p.add_argument('--model-config',type=Path,help='Additional Diffusers model profiles in JSON')
    p.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    p.add_argument('--model-path',type=Path,help='Local checkpoint override; its content is still pinned')
    p.add_argument('--pup-path',type=Path,help='FLUX3-only local official prompt-upsampler directory override')
    p.add_argument('--output-dir',type=Path,required=True)
    scope=p.add_mutually_exclusive_group(required=True);scope.add_argument('--ids');scope.add_argument('--all',action='store_true')
    p.add_argument('--run-id',help='Stable label within this model, defaults to output directory name')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,help='Explicit sampling override, stored in the protocol')
    p.add_argument('--cpu-offload',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--repair-invalid',action=argparse.BooleanOptionalAction,default=True,
                   help='Recreate invalid completed outputs in a new attempt; default enabled')
    p.add_argument('--device',default='cuda',help='Allocated CUDA device, e.g. cuda or cuda:0')
    p.add_argument('--require-h200',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Validate/hash local inputs and checkpoint without torch or model loading')
    return p


def main(argv=None):
    args=parser().parse_args(argv);load_model_config(args.model_config);config=get_model_config(args.model)
    args.dataset_root=args.dataset_root.resolve();args.output_dir=args.output_dir.resolve()
    args.model_path=(args.model_path or Path(config['local_dir'])).resolve()
    if args.pup_path is not None:
        if config.get('backend')!='flux3-modular-local-pup':raise ValueError('--pup-path is only valid for FLUX3')
        config['dependencies'][0]['local_dir']=str(args.pup_path.resolve())
    if args.output_dir in (args.dataset_root,args.model_path):raise ValueError('Use a dedicated generation run directory')
    if not re.fullmatch(r'cuda(?::[0-9]+)?',args.device):raise ValueError('Local generation requires an allocated CUDA device')
    if not 0<=args.seed<2**63:raise ValueError('Seed must be a nonnegative signed64-bit integer')
    if args.steps is not None:
        if args.steps<1:raise ValueError('Steps must be positive')
        config['num_inference_steps']=args.steps
    args.cpu_offload=config['cpu_offload'] if args.cpu_offload is None else args.cpu_offload
    if args.run_id is not None and (not args.run_id.strip() or args.run_id!=args.run_id.strip()):raise ValueError('Invalid run ID')
    with generation_lock(args.output_dir):
        require_directory(args.output_dir)
        tasks=select_tasks(args.dataset_root,args.ids,args.all)
        validate_model_files(args.model_path,config);validate_pipeline_signature(config)
        plan=input_plan(args.dataset_root,tasks)
        torch=None
        if not args.preflight:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; submit a GPU job or use --preflight')
            gpu=torch.cuda.get_device_name(args.device)
            if args.require_h200 and 'H200' not in gpu.upper():raise RuntimeError(f'Expected H200 allocation, got {gpu}')
            torch.set_num_threads(8)
        print(json.dumps({'status':'hashing_model_content','model':args.model,'model_path':str(args.model_path)}),flush=True)
        if config.get('backend')=='flux3-modular-local-pup':
            from flux3_adapter import collect_flux3_content
            content=collect_flux3_content(args.model_path,config)
        else:content=collect_generation_content(args.model_path)
        protocol=build_protocol(args,config,tasks,plan,content)
        records=require_directory(args.output_dir,protocol)
        write(args.output_dir/'protocol.json',protocol);write(args.output_dir/'input_plan.json',plan)
        if args.preflight:
            print(json.dumps({'status':'preflight_passed','model':args.model,'cases':len(tasks),
                              'protocol_sha256':protocol['protocol_sha256'],'output_dir':str(args.output_dir)}),flush=True)
            return 0
        return run_generation(args,config,protocol,tasks,plan,records,torch)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'Generation failed: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)
