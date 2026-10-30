"""Download pinned Diffusers checkpoints using the existing Hugging Face login.

No credentials are accepted on the command line or written to progress files.
The downloader omits duplicate root single-file weights when component folders
are used, and validates the required components and every checkpoint shard.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import re
import tempfile
import threading

from settings import MODEL_ROOT
from model_registry import MODEL_CONFIGS,get_model_config,validate_model_files,load_model_config
try:
    from model_registry import DEFAULT_MODELS
except ImportError:
    DEFAULT_MODELS=tuple(MODEL_CONFIGS)


def now():return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',mode='w',delete=False) as stream:
            temporary=Path(stream.name);json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


@contextmanager
def model_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('A download for this model is already running') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def selected_models(value):
    if value=='all':return list(DEFAULT_MODELS)
    result=[part.strip() for part in value.split(',')]
    if not result or any(not part for part in result) or len(set(result))!=len(result):
        raise ValueError('--models must be all or unique comma-separated model IDs')
    for model_id in result:get_model_config(model_id)
    return result


def acquisition_plans(models):
    plans={}
    for model_id in models:
        config=get_model_config(model_id)
        for dependency in config.get('dependencies',[]):
            dependency=dict(dependency,backend='transformers-pup')
            plans.setdefault(dependency['id'],dependency)
        plans.setdefault(model_id,config)
    return plans


def select_diffusers_files(info,config):
    backend=config.get('backend','diffusion-pipeline')
    components=set(config.get('required_components',[]))
    root_files={'model_index.json'}
    if backend=='flux3-modular-local-pup':
        components={'text_encoder','transformer','vae'}
        root_files={'modular_model_index.json','config.json','manifest.json','README.md','LICENSE.md','NOTICE.md'}
    elif backend=='transformers-pup':
        components=set();root_files=None
    files=[]
    for sibling in info.siblings:
        name=sibling.rfilename
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe repository file path')
        selected=(root_files is None and not name.startswith('.')) or (root_files is not None and name in root_files) or ('/' in name and name.split('/',1)[0] in components)
        if selected:
            if not isinstance(sibling.size,int) or sibling.size<0:raise ValueError('Repository metadata has no exact file size')
            lfs=getattr(sibling,'lfs',None)
            files.append({'path':name,'bytes':sibling.size,
                          'lfs_sha256':getattr(lfs,'sha256',None) if lfs is not None else None})
    names={item['path'] for item in files}
    required={'model_index.json'} if backend=='diffusion-pipeline' else {'modular_model_index.json'} if backend=='flux3-modular-local-pup' else {'config.json','model.safetensors','system_prompts.json'}
    if not required.issubset(names):raise ValueError('Repository is missing its required model/processor metadata')
    for component in components:
        if not any(name.startswith(component+'/') for name in names):
            raise ValueError('Repository has no required Diffusers component: '+component)
    return sorted(files,key=lambda item:item['path'])


def local_progress(folder,files):
    complete=0;completed_bytes=0;statuses={}
    for item in files:
        path=folder/item['path']
        try:correct=path.is_file() and path.stat().st_size==item['bytes']
        except OSError:correct=False
        if correct:complete+=1;completed_bytes+=item['bytes']
        statuses[item['path']]='complete' if correct else 'pending_or_partial'
    return {'complete_files':complete,'expected_files':len(files),'complete_bytes':completed_bytes,
            'expected_bytes':sum(item['bytes'] for item in files),'file_status':statuses}


def validate_download(folder,files,config):
    progress=local_progress(folder,files)
    if progress['complete_files']!=progress['expected_files']:
        raise ValueError('Downloaded file inventory is incomplete or contains wrong sizes')
    if config.get('backend')=='transformers-pup':
        index=json.loads((folder/'config.json').read_text())
        if 'Qwen3_5ForConditionalGeneration' not in index.get('architectures',[]):
            raise ValueError('Prompt upsampler architecture mismatch')
        if not isinstance(json.loads((folder/'system_prompts.json').read_text()),dict):
            raise ValueError('Prompt upsampler system prompts must be a JSON object')
        index={'_class_name':'Qwen3_5ForConditionalGeneration'}
    else:index=validate_model_files(folder,config)
    names={item['path'] for item in files}
    for item in files:
        name=item['path']
        if not name.endswith('.index.json'):continue
        data=json.loads((folder/name).read_text())
        mapping=data.get('weight_map') if isinstance(data,dict) else None
        if mapping is None:continue
        if not isinstance(mapping,dict):raise ValueError('Invalid checkpoint weight_map')
        for shard in mapping.values():
            if not isinstance(shard,str):raise ValueError('Invalid checkpoint shard reference')
            target=Path(name).parent/shard
            if target.is_absolute() or '..' in target.parts or target.as_posix() not in names:
                raise ValueError('Checkpoint index references a missing/unsafe shard')
    return {'pipeline':index.get('_class_name',config.get('pipeline')),'file_inventory_complete':True,'shard_references_complete':True,
            'integrity_note':'Pinned Hugging Face snapshot transfer and exact file sizes; generation separately computes actual content SHA-256.'}


def safe_error(exc):
    status=getattr(getattr(exc,'response',None),'status_code',None)
    return {'error_type':type(exc).__name__,'http_status':status}


def download_one(model_id,output_root,status_root,*,api,token,metadata_only=False,max_workers=4,snapshot=None,head=None,config_override=None):
    config=dict(config_override or get_model_config(model_id));folder=Path(output_root)/Path(config['local_dir']).name
    config['local_dir']=str(folder.resolve())
    config['dependencies']=[{**dep,'local_dir':str((Path(output_root)/Path(dep['local_dir']).name).resolve())}
                            for dep in config.get('dependencies',[])]
    status_path=Path(status_root)/(model_id+'.json')
    state={'model_id':model_id,'repo_id':config['repo_id'],'local_dir':str(folder.resolve()),
           'status':'checking','metadata_only':metadata_only,'role':config.get('role','benchmark_model'),'updated_at':now()}
    def save(**updates):
        state.update(updates,updated_at=now());write_json(status_path,state)
    with model_lock(status_path.with_suffix('.lock')):
        previous=json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous and (previous.get('repo_id')!=config['repo_id'] or previous.get('local_dir')!=str(folder.resolve())):
            raise ValueError('Existing download state belongs to another model or output directory')
        revision=previous.get('revision')
        save()
        try:
            if config.get('prefer_existing') and folder.exists() and any(folder.iterdir()):
                # Existing benchmark checkpoints are local historical artifacts.
                # Reuse them after component/shard validation without assigning
                # today's remote commit or overwriting their weight files.
                files=[]
                for path in sorted(folder.rglob('*')):
                    if not path.is_file() or '.cache' in path.parts or '__pycache__' in path.parts:continue
                    files.append({'path':path.relative_to(folder).as_posix(),'bytes':path.stat().st_size,'lfs_sha256':None})
                validation=validate_download(folder,files,config)
                validation['integrity_note']='Existing local component/shard/size validation; no remote commit assigned. Generation separately pins actual local content hashes.'
                save(status='complete',phase='complete',revision=None,files=files,
                     acquisition='reused_existing_local',local_commit_verified=False,
                     validation=validation,progress=local_progress(folder,files))
                return dict(state)
            # A retry keeps the first resolved commit, including after an
            # access approval; a moving main branch cannot mix checkpoint parts.
            kwargs={'files_metadata':True}
            if revision:kwargs['revision']=revision
            info=api.model_info(config['repo_id'],**kwargs)
            if not isinstance(info.sha,str) or not re.fullmatch('[0-9a-f]{40}',info.sha):
                raise ValueError('Repository metadata did not resolve an immutable commit SHA')
            files=select_diffusers_files(info,config)
            state.update(revision=info.sha,gated=info.gated,files=files,
                         duplicate_root_weight_files_excluded=config.get('backend','diffusion-pipeline')!='transformers-pup')
            state['phase']='access_check'
            backend=config.get('backend','diffusion-pipeline')
            metadata_name=('modular_model_index.json' if backend=='flux3-modular-local-pup'
                           else 'config.json' if backend=='transformers-pup' else 'model_index.json')
            if head is not None:head(config['repo_id'],metadata_name,info.sha,token)
            existing=local_progress(folder,files)
            save(status='ready' if metadata_only else 'downloading',phase='metadata_complete' if metadata_only else 'download',progress=existing)
            if metadata_only:return dict(state)
            if snapshot is None:
                from huggingface_hub import snapshot_download
                snapshot=snapshot_download
            stopped=threading.Event()
            def monitor():
                while not stopped.wait(10):
                    save(progress=local_progress(folder,files))
            watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
            try:
                snapshot(repo_id=config['repo_id'],revision=info.sha,local_dir=str(folder),token=token,
                         allow_patterns=[item['path'] for item in files],max_workers=max_workers)
            finally:
                stopped.set();watcher.join()
            save(phase='validation',progress=local_progress(folder,files))
            validation=validate_download(folder,files,config)
            save(status='complete',phase='complete',progress=local_progress(folder,files),validation=validation)
        except Exception as exc:
            details=safe_error(exc)
            gated=details['http_status'] in (401,403)
            save(status='needs_access' if gated else 'error',**details,
                 access_url='https://huggingface.co/'+config['repo_id'] if gated else None)
        return dict(state)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='all',help='all or comma-separated registered model IDs')
    p.add_argument('--model-config',type=Path,help='Additional Diffusers model profiles in JSON')
    p.add_argument('--output-root',type=Path,default=MODEL_ROOT)
    p.add_argument('--status-dir',type=Path,help='Progress directory; defaults to OUTPUT_ROOT/.model_downloads')
    p.add_argument('--metadata-only',action='store_true',help='Check login/access/revision and sizes; do not download weights')
    p.add_argument('--max-workers',type=int,default=4)
    a=p.parse_args(argv)
    if not 1<=a.max_workers<=16:p.error('--max-workers must be between 1 and 16')
    try:
        load_model_config(a.model_config)
        models=selected_models(a.models)
    except ValueError as exc:p.error(str(exc))
    from huggingface_hub import HfApi,get_token,get_hf_file_metadata,hf_hub_url
    token=get_token();api=HfApi(token=token)
    def head(repo,filename,revision,credential):
        get_hf_file_metadata(hf_hub_url(repo,filename,revision=revision),token=credential)
    status_root=a.status_dir or a.output_root/'.model_downloads'
    reports=[]
    plans=acquisition_plans(models)
    for model_id,config in plans.items():
        try:
            result=download_one(model_id,a.output_root,status_root,api=api,token=token,
                metadata_only=a.metadata_only,max_workers=a.max_workers,head=head,config_override=config)
        except Exception as exc:
            result={'model_id':model_id,'status':'error',**safe_error(exc)}
        reports.append(result)
        short={key:result[key] for key in ('model_id','status','revision','access_url','error_type','http_status') if key in result}
        if 'progress' in result:short['progress']={key:value for key,value in result['progress'].items() if key!='file_status'}
        print(json.dumps(short),flush=True)
    complete=all(report['status'] in ('ready','complete') for report in reports)
    print(json.dumps({'complete':complete,'benchmark_models':len(models),'acquisitions_including_dependencies':len(plans),'metadata_only':a.metadata_only,
                      'status_dir':str(status_root.resolve())}),flush=True)
    return 0 if complete else 1


if __name__=='__main__':raise SystemExit(main())
