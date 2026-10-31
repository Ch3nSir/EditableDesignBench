"""TextEditBench MLLM criteria adapter; no scoring is invented without an API.

Default: no-reference evaluation, using source + prediction + user instruction.
The original MIT-licensed IF/TA/VC/LP(/SE) prompts are snapshotted verbatim in
mllm_prompts.json; reference mode is explicit and separately identified.
"""
from __future__ import annotations

from settings import DATASET_ROOT, MODEL_ROOT

import argparse
import base64
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from threading import Event, Lock
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = Path(__file__).with_name('mllm_prompts.json')
BUNDLED_EVALUATION = ROOT/'vendor/texteditbench/evaluation'
STANDARD_DIMENSIONS = ('IF', 'TA', 'VC', 'LP')
METHODS = {'delete':'text_delete', 'insert':'text_insert', 'replace':'text_change',
           'move':'text_relocation', 'scale':'scaling', 'color':'text_attribute'}
PROTOCOL_VERSION = 'texteditbench-mllm-adapter-v2'
CACHE_SCHEMA_VERSION = 1


class JudgeError(Exception):
    """Safe-to-report failure; provider bodies/credentials are never echoed."""


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_prompt_source(item: dict) -> Path:
    """Locate an original prompt source and verify its snapshotted bytes.

    Prefer the recorded path whenever it remains available, preserving local
    paid-score fingerprints. A relocated installation uses the bundled copy.
    The file is read only; its import-time client setup is never executed.
    """
    source=Path(item['source_file'])
    if not source.is_absolute():source=ROOT/source
    if not source.is_file():source=BUNDLED_EVALUATION/source.name
    if not source.is_file():raise JudgeError('Neither original nor bundled official prompt source is available')
    if digest(source.read_bytes())!=item.get('source_sha256'):
        raise JudgeError('Official prompt source SHA-256 differs from the snapshot')
    return source


def canonical_hash(value) -> str:
    return digest(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())


def write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace one file; simultaneous writers never share a temp."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name)
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)


def write_json(path: Path, obj) -> None:
    write_bytes(path,(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


def cache_path(output: Path) -> Path:
    return output.with_name(output.name+'.cache.json')


@contextmanager
def output_lock(output: Path):
    """Hold one advisory lock for the whole read/merge/request/write cycle."""
    output.parent.mkdir(parents=True,exist_ok=True)
    lock_path=output.with_name(output.name+'.lock')
    with lock_path.open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise JudgeError('This output already has an active evaluation; use a separate output or wait') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def merge_cached_scores(cache: dict, records: list[dict]) -> None:
    """Keep only validated paid dimensions; unavailable inputs never erase them."""
    if not isinstance(records,list):raise JudgeError('Invalid cached records list')
    for record in records:
        if not isinstance(record,dict):raise JudgeError('Invalid cached record')
        fingerprint=record.get('fingerprint')
        if not fingerprint:continue  # Pending/error placeholders have no scores.
        if not isinstance(record.get('id'),str) or not isinstance(fingerprint,str) or not re.fullmatch('[0-9a-f]{64}',fingerprint):
            raise JudgeError('Invalid cached case identity')
        dimensions=record.get('dimensions',{})
        if not isinstance(dimensions,dict):raise JudgeError('Invalid cached dimensions')
        scored={}
        for dimension,value in dimensions.items():
            if not isinstance(value,dict):raise JudgeError('Invalid cached dimension')
            if value.get('status')!='scored':continue
            if dimension not in (*STANDARD_DIMENSIONS,'SE'):raise JudgeError('Unknown cached score dimension')
            parse_score(json.dumps({dimension:value.get('score'),'rationale':value.get('rationale')}),dimension)
            scored[dimension]=dict(value)
        if not scored:continue
        key=(record['id'],fingerprint)
        if key not in cache:cache[key]=dict(record,dimensions={})
        cache[key]['dimensions'].update(scored)


def load_score_cache(output: Path) -> dict:
    """Import the legacy in-report cache and the persistent sidecar safely."""
    cache={}
    sidecar=cache_path(output)
    if output.exists():
        saved=json.loads(output.read_text())
        if not isinstance(saved,dict) or not isinstance(saved.get('results'),list):
            raise JudgeError('Invalid existing evaluation report; restore its backup before continuing')
        merge_cached_scores(cache,saved.get('_resume_cache',[]))
        merge_cached_scores(cache,saved['results'])
    # The sidecar is committed first, so it may contain a newer paid response
    # than an interrupted report write. It must win duplicate dimensions.
    if sidecar.exists():
        saved=json.loads(sidecar.read_text())
        if not isinstance(saved,dict) or saved.get('schema_version')!=CACHE_SCHEMA_VERSION:
            raise JudgeError('Unsupported or invalid score cache schema')
        merge_cached_scores(cache,saved.get('records'))
    return cache


def backup_existing(path: Path) -> None:
    if path.exists():write_bytes(path.with_name(path.name+'.bak'),path.read_bytes())


def load_tasks(path: Path, ids: str | None = None) -> list[dict]:
    tasks=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen=set()
    for t in tasks:
        if not isinstance(t,dict) or not isinstance(t.get('id'),str) or not isinstance(t.get('instruction'),str):
            raise JudgeError('Task entries require string id and instruction')
        if t['id'] in seen:raise JudgeError(f'Duplicate task ID: {t["id"]}')
        if t.get('operation') not in METHODS:raise JudgeError(f'Unsupported operation: {t["id"]}')
        seen.add(t['id'])
    if ids is not None:
        requested=[s.strip() for s in ids.split(',')]
        if any(not s for s in requested):raise JudgeError('Empty ID token in --ids')
        if len(requested)!=len(set(requested)):raise JudgeError('Duplicate requested ID in --ids')
        unknown=sorted(set(requested)-seen)
        if not requested or unknown:raise JudgeError('Unknown/empty --ids: '+','.join(unknown))
        tasks=[t for t in tasks if t['id'] in set(requested)]
    if not tasks:raise JudgeError('No tasks selected')
    return tasks


def request_parameters(config: dict) -> dict:
    """Keep official legacy defaults; opt in explicitly for reasoning models."""
    profile=config.get('request_profile','texteditbench')
    if profile=='texteditbench':
        parameters={'temperature':0.0,'max_tokens':500}
    elif profile=='reasoning_chat':
        limit=config.get('max_completion_tokens',4096)
        effort=config.get('reasoning_effort','low')
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=128000:
            raise JudgeError('max_completion_tokens must be an integer in [1,128000]')
        if effort not in ('none','minimal','low','medium','high','xhigh','max'):
            raise JudgeError('Invalid reasoning_effort')
        parameters={'max_completion_tokens':limit,'reasoning_effort':effort}
    else:raise JudgeError('Unknown request_profile; use texteditbench or reasoning_chat')
    if 'store' in config:
        if not isinstance(config['store'],bool):raise JudgeError('store must be a boolean')
        parameters['store']=config['store']
    return parameters


def load_credentials(args) -> dict:
    config={}
    if args.config:
        config=json.loads(args.config.read_text())
        if not isinstance(config,dict):raise JudgeError('MLLM config must be a JSON object')
    envname=args.api_key_env or config.get('api_key_env')
    if envname:
        key=os.getenv(envname,'')
    else:
        key=next((os.environ[n] for n in ('MLLM_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY') if os.getenv(n)), '')
    keyfile=config.get('api_key_file')
    if not key and keyfile:
        p=Path(keyfile).expanduser()
        if not p.is_absolute() and args.config:p=args.config.parent/p
        key=p.read_text().strip()
    # Credentials may be supplied in a user-owned config; never persist config
    # contents or the credential in reports or request fingerprints.
    if not key:key=config.get('api_key','')
    if key and (not isinstance(key,str) or '\n' in key or '\r' in key):raise JudgeError('Invalid API credential format')
    base=(args.base_url or config.get('base_url') or os.getenv('MLLM_BASE_URL')
          or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_URL') or 'https://api.openai.com/v1').rstrip('/')
    u=urllib.parse.urlsplit(base)
    if u.scheme not in {'https','http'} or not u.netloc or u.username or u.password or u.query or u.fragment:
        raise JudgeError('API base URL must be an http(s) endpoint without embedded credentials/query/fragment')
    endpoint=base if base.endswith('/chat/completions') else base+'/chat/completions'
    model=args.model or config.get('model') or os.getenv('MLLM_MODEL') or os.getenv('OPENAI_MODEL')
    if model is not None and (not isinstance(model,str) or not model.strip()):raise JudgeError('Invalid judge model')
    return {'api_key':key,'endpoint':endpoint,'model':model,
            'timeout':float(config.get('timeout',args.timeout)),
            'max_retries':int(config.get('max_retries',args.max_retries)),
            'request_parameters':request_parameters(config)}


def image_data_url(array: np.ndarray) -> str:
    if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=3:
        raise JudgeError('MLLM image must be a uint8 RGB array')
    out=io.BytesIO();Image.fromarray(array).save(out,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode('ascii')


def build_messages(dimension: str, system_prompt: str, task: dict,
                   source_url: str, prediction_url: str, reference_url: str | None = None) -> list[dict]:
    content=[{'type':'text','text':f'User editing instruction (prompt): {task["instruction"]}'}]
    # Match official callers: no-reference supplies method for IF only;
    # reference mode supplies it on all dimensions.
    if dimension=='IF' or reference_url is not None:
        content.append({'type':'text','text':f'Editing method: {METHODS[task["operation"]]}'} )
    if dimension=='SE':content.append({'type':'text','text':'Knowledge prompt: '+task['knowledge_prompt']})
    content.extend([{'type':'text','text':'Input image (original):'},
                    {'type':'image_url','image_url':{'url':source_url}},
                    {'type':'text','text':'Pred image (model output):'},
                    {'type':'image_url','image_url':{'url':prediction_url}}])
    if reference_url is not None:
        content.extend([{'type':'text','text':'Output image (expected reference):'},
                        {'type':'image_url','image_url':{'url':reference_url}}])
    # Preserve each official mode's complete text, including its JSON footer.
    # Changing this request template also requires a protocol version change so
    # scores obtained with a previous prompt cannot be reused by resume.
    if reference_url is None:
        footer=('\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object in this exact format:**\n'
                f'{{"{dimension}": <score 0-5>, "rationale": "<explanation>"}}\n'
                '**NO markdown, NO code blocks, NO explanations outside the JSON, NO think tags. Just the raw JSON object.**')
    else:
        footer=('\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object.**\n'
                f'{{"{dimension}": <score 0-5>, "rationale": "<explanation>"}}')
    content.append({'type':'text','text':footer})
    return [{'role':'system','content':system_prompt},{'role':'user','content':content}]


def parse_score(content, dimension: str) -> dict:
    if isinstance(content,list):
        content=''.join(p.get('text','') for p in content if isinstance(p,dict) and p.get('type')=='text')
    if not isinstance(content,str):raise JudgeError('Judge response content is not text')
    text=content.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I)
        text=re.sub(r'\s*```$','',text)
    try:data=json.loads(text)
    except (ValueError,TypeError) as exc:raise JudgeError('Judge returned invalid JSON') from exc
    if not isinstance(data,dict) or dimension not in data:raise JudgeError('Judge JSON is missing score')
    value=data[dimension]
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=5:
        raise JudgeError('Judge score must be numeric in [0,5]')
    rationale=data.get('rationale')
    if not isinstance(rationale,str) or not rationale.strip():raise JudgeError('Judge JSON is missing rationale')
    return {dimension:value,'rationale':rationale}


def post_json(endpoint: str, key: str, payload: dict, timeout: float) -> dict:
    req=urllib.request.Request(endpoint,data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    except urllib.error.HTTPError as exc:
        # Avoid echoing arbitrary provider error text, request images or secrets.
        raise JudgeError(f'API HTTP {exc.code}') from None
    except (urllib.error.URLError,TimeoutError,OSError):raise JudgeError('API connection/timeout failure') from None
    except ValueError:raise JudgeError('API returned invalid response JSON') from None


def request_score(dimension: str, messages: list[dict], credentials: dict,
                  transport: Callable = post_json, sleeper: Callable = time.sleep) -> dict:
    retries=credentials['max_retries']
    if not 0<=retries<=10 or credentials['timeout']<=0:raise JudgeError('Invalid retry/timeout configuration')
    last='Judge request failed'
    for attempt in range(retries+1):
        try:
            data=transport(credentials['endpoint'],credentials['api_key'],
                {'model':credentials['model'],'messages':messages,
                 **credentials.get('request_parameters',request_parameters({}))},credentials['timeout'])
            choice=data['choices'][0]
            if choice.get('finish_reason')=='length':raise JudgeError('Judge response hit token limit')
            result=parse_score(choice['message']['content'],dimension)
            scored={'status':'scored','score':result[dimension],'rationale':result['rationale'],'attempts':attempt+1}
            usage=data.get('usage')
            if isinstance(usage,dict):
                scored['usage']={name:value for name in ('prompt_tokens','completion_tokens','total_tokens')
                                 if isinstance((value:=usage.get(name)),int) and not isinstance(value,bool) and value>=0}
            if isinstance(data.get('model'),str):scored['returned_model']=data['model']
            return scored
        except (JudgeError,KeyError,IndexError,TypeError) as exc:
            last=str(exc) if isinstance(exc,JudgeError) else 'Malformed API response'
            if attempt<retries:sleeper(min(2**attempt,20))
    return {'status':'error','score':None,'rationale':None,'error':last,'attempts':retries+1}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode!='RGB':raise JudgeError('MLLM image must be the validated RGB export')
        return np.asarray(image).copy()


def load_manifest(path: Path | None):
    if path is None:return None
    from output_adapter import load_generation_records
    return load_generation_records(path)


def prepare_case(task: dict, args, manifest, protocol: dict) -> tuple[dict, dict]:
    source_path=(args.dataset/task['image']).resolve()
    source=read_rgb(source_path)
    if args.generation_manifest is not None:
        from output_adapter import load_prediction
        record=manifest.get(task['id'])
        if record is None:raise JudgeError('Generation manifest is missing case ID')
        prediction,provenance=load_prediction(task,args.dataset,record=record,manifest_path=args.generation_manifest)
    else:
        if args.predictions is None:raise JudgeError('Specify --predictions or --generation-manifest')
        pred_path=(args.predictions/task.get('output_filename',task['id']+'.png')).resolve()
        prediction=read_rgb(pred_path)
        provenance={'rgb_output':str(pred_path),'rgb_sha256':digest(pred_path.read_bytes()),
                    'evaluation_roi':[0,0,source.shape[1],source.shape[0]]}
    if source.shape!=prediction.shape:raise JudgeError('Source and prediction evaluation ROI dimensions differ')
    source_url=image_data_url(source);prediction_url=image_data_url(prediction)
    reference=None;reference_hash=None
    if args.mode=='reference':
        ref_path=args.dataset/'cases'/task['id']/'target.png'
        reference_array=read_rgb(ref_path)
        if reference_array.shape!=source.shape:raise JudgeError('Reference dimensions differ')
        reference=image_data_url(reference_array);reference_hash=digest(reference_array.tobytes())
    knowledge=task.get('knowledge_prompt')
    if knowledge is not None and (not isinstance(knowledge,str) or not knowledge.strip()):knowledge=None
    dimensions=list(STANDARD_DIMENSIONS)+(['SE'] if knowledge else [])
    # Preserve the paid-score identity when only validation diagnostics are
    # added to provenance. The image bytes/pixels, instruction and judge
    # protocol remain independently bound below and are revalidated first.
    legacy_fields=(('mode','manifest','raw_output','raw_sha256','rgb_output','rgb_sha256',
                    'source_size','canvas_size','evaluation_roi','padding','adapter_version')
                   if args.generation_manifest is not None else ('rgb_output','rgb_sha256','evaluation_roi'))
    identity_provenance={key:provenance[key] for key in legacy_fields if key in provenance}
    signature=canonical_hash({'source_sha256':digest(source_path.read_bytes()),
        'prediction_pixels_sha256':digest(prediction.tobytes()),'reference_pixels_sha256':reference_hash,
        'provenance':identity_provenance,'instruction':task['instruction'],'editing_method':METHODS[task['operation']],
        'knowledge_prompt':knowledge,'protocol_sha256':canonical_hash(protocol)})
    record={'id':task['id'],'source_id':task.get('source_id'),'domain':task.get('domain'),
        'operation':task['operation'],'instruction':task['instruction'],'editing_method':METHODS[task['operation']],
        'fingerprint':signature,'status':'not_scored','prediction_provenance':provenance,
        'dimensions':{d:{'status':'not_scored','score':None,'rationale':None} for d in dimensions}}
    if not knowledge:record['dimensions']['SE']={'status':'not_applicable','score':None,'reason':'No explicit knowledge_prompt; follows official optional SE rule.'}
    urls={'source':source_url,'prediction':prediction_url,'reference':reference}
    return record,urls


def summarize(records: list[dict]) -> dict:
    result={'expected':len(records),'scored':sum(r['status']=='scored' for r in records),
            'pending':sum(r['status']=='not_scored' for r in records),
            'errors':sum(r['status'] in {'error','partial_error'} for r in records),'criteria':{}}
    for d in (*STANDARD_DIMENSIONS,'SE'):
        applicable=[r['dimensions'][d] for r in records if d in r.get('dimensions',{}) and r['dimensions'][d]['status']!='not_applicable']
        scores=[v['score'] for v in applicable if v['status']=='scored']
        # An aggregate score is publishable only on the full applicable set.
        # Keep partial per-case scores/coverage without silently ranking subsets.
        complete=bool(applicable) and len(scores)==len(applicable)
        result['criteria'][d]={'mean':sum(scores)/len(scores) if complete else None,
            'scored':len(scores),'expected':len(applicable),'coverage':len(scores)/len(applicable) if applicable else None}
    return result


def unprocessed_record(task: dict) -> dict:
    dimensions={d:{'status':'not_scored','score':None,'rationale':None,'reason':'not_processed'}
                for d in STANDARD_DIMENSIONS}
    knowledge=task.get('knowledge_prompt')
    dimensions['SE']=({'status':'not_scored','score':None,'rationale':None,'reason':'not_processed'}
        if isinstance(knowledge,str) and knowledge.strip()
        else {'status':'not_applicable','score':None,'reason':'No explicit knowledge_prompt'})
    return {'id':task['id'],'operation':task['operation'],'status':'not_scored','dimensions':dimensions}


def evaluate(args, transport: Callable = post_json, sleeper: Callable = time.sleep) -> tuple[dict,int]:
    args.output=args.output.resolve()
    with output_lock(args.output):
        return _evaluate_locked(args,transport,sleeper)


def _evaluate_locked(args, transport: Callable, sleeper: Callable) -> tuple[dict,int]:
    workers=getattr(args,'workers',5)
    if type(workers) is not int or not 1<=workers<=32:
        raise JudgeError('--workers must be an integer in [1,32]')
    tasks=load_tasks(args.tasks or args.dataset/'tasks.jsonl',args.ids)
    prompt_data=json.loads(PROMPT_FILE.read_text())
    prompt_item=prompt_data['modes'][args.mode]
    prompt_source=resolve_prompt_source(prompt_item)
    prompts=prompt_item['prompts']
    credentials=load_credentials(args)
    protocol={'version':PROTOCOL_VERSION,'mode':args.mode,'judge_model':credentials['model'],
        'endpoint':credentials['endpoint'],'prompts_sha256':digest(PROMPT_FILE.read_bytes()),
        'prompt_source':str(prompt_source),
        'prompt_source_sha256':prompt_data['modes'][args.mode]['source_sha256'],
        'user_message_template':'Verbatim official mode-specific text; validated RGB PNG image URLs',
        'criteria':list(STANDARD_DIMENSIONS),'optional_SE':'Only explicit knowledge_prompt',
        'score_range':[0,5],'temperature':0,'max_tokens_per_criterion':500,
        'separate_request_per_criterion':True,'independent_case_context':True,
        'images':'source + prediction ROI' if args.mode=='no_reference' else 'source + prediction ROI + reference',
        'image_encoding':'Lossless RGB PNG; padding removed only from in-memory arrays; files unchanged',
        'operation_mapping':METHODS,'API_failure_scores':'null; never fabricated zero',
        'aggregate_means':'null unless every applicable expected case has a scored result',
        'API_connectivity_validated_by_dry_run':False}
    parameters=credentials['request_parameters']
    if parameters!=request_parameters({}):
        # Leave old official fingerprints identical. Explicit alternate API
        # parameters participate in input/cache and cross-model comparison IDs.
        protocol.update(request_parameters=parameters,temperature=parameters.get('temperature'),
                        max_tokens_per_criterion=parameters.get('max_tokens',parameters.get('max_completion_tokens')))
    old=load_score_cache(args.output)
    manifest=load_manifest(args.generation_manifest)
    # Validate everything that can fail before replacing the current report.
    # One recoverable snapshot per run avoids a backup per dimension/request.
    backup_existing(args.output)
    backup_existing(cache_path(args.output))
    pending_reason='dry_run' if args.dry_run else ('needs_api' if not credentials['api_key'] or not credentials['model'] else None)
    report={'protocol':protocol,'status':'not_scored' if pending_reason else 'running',
            'pending_reason':pending_reason,'configuration_missing':
                [name for name in ('api_key','model') if not credentials[name]],
            'api_calls_made':0,'summary':{},'results':[unprocessed_record(t) for t in tasks],
            'storage':{'cache_schema_version':CACHE_SCHEMA_VERSION,
                       'cache_file':str(cache_path(args.output)),
                       'cache_scope':'Validated paid scores for all historical IDs and input/judge fingerprints; excluded from this report denominator'}}
    positions={r['id']:i for i,r in enumerate(report['results'])}
    state_lock=Lock();stop=Event()
    counts={'done':0,'failed':0,'cached':0,'cases':0}
    report['resume']={'enabled':args.resume,'cached_dimensions':0}
    if args.resume and old and getattr(args,'progress',True):
        available=sum(len(record['dimensions']) for record in old.values())
        print(f'MLLM resume: {available} successful cached dimensions found; validating inputs before reuse.',file=sys.stderr,flush=True)
    total_dimensions=sum(len(STANDARD_DIMENSIONS)+(isinstance(task.get('knowledge_prompt'),str)
                         and bool(task['knowledge_prompt'].strip())) for task in tasks)
    report['execution']={'workers':workers,'concurrency_unit':'case','maximum_simultaneous_api_requests':workers}
    def counted_transport(*values):
        with state_lock:
            if stop.is_set():raise JudgeError('Evaluation interrupted')
            report['api_calls_made']+=1
        return transport(*values)
    def checkpoint():
        # Caller holds state_lock: response mutation, cache writes and report
        # writes are serialized. Network requests and retry waits are outside it.
        # Persist the paid response before its report checkpoint. A subsequent
        # report I/O failure or interruption can then resume without re-paying.
        report['resume']['cached_dimensions']=counts['cached']
        merge_cached_scores(old,report['results'])
        write_json(cache_path(args.output),{'schema_version':CACHE_SCHEMA_VERSION,'records':list(old.values())})
        report['summary']=summarize(report['results']);write_json(args.output,report)
    def safe_error(exc):
        text=str(exc) if isinstance(exc,(JudgeError,FileNotFoundError,ValueError)) else type(exc).__name__
        return text.replace(credentials['api_key'],'[REDACTED]') if credentials['api_key'] else text

    def failed_case(task,exc):
        with state_lock:
            existing=report['results'][positions[task['id']]]
            existing.update(status='error',error=safe_error(exc))
            for value in existing['dimensions'].values():
                if value['status']=='not_scored':
                    value.update(status='error',error=safe_error(exc))
                    counts['done']+=1;counts['failed']+=1
            counts['cases']+=1
            checkpoint()

    def evaluate_case(task):
        if stop.is_set():return
        try:
            case,urls=prepare_case(task,args,manifest,protocol)
        except Exception as exc:
            failed_case(task,exc);return
        with state_lock:
            previous=deepcopy(old.get((case['id'],case['fingerprint']),{})) if args.resume else {}
            if previous.get('fingerprint')==case['fingerprint']:
                for d,v in case['dimensions'].items():
                    cached=previous.get('dimensions',{}).get(d,{})
                    if cached.get('status')=='scored':
                        parse_score(json.dumps({d:cached.get('score'),'rationale':cached.get('rationale')}),d)
                        case['dimensions'][d]=cached
                        counts['done']+=1;counts['cached']+=1
            report['results'][positions[case['id']]]=case
            checkpoint()
        for dimension,value in case['dimensions'].items():
            if stop.is_set():break
            if value['status'] in {'not_applicable','scored'}:continue
            if pending_reason:
                with state_lock:
                    value['reason']=pending_reason
                    counts['done']+=1;checkpoint()
            else:
                try:
                    messages=build_messages(dimension,prompts[dimension],task,urls['source'],urls['prediction'],urls['reference'])
                    result=request_score(dimension,messages,credentials,counted_transport,sleeper)
                except Exception as exc:
                    failed_case(task,exc);return
                with state_lock:
                    case['dimensions'][dimension]=result
                    counts['done']+=1;counts['failed']+=result['status']=='error'
                    # Persist each paid response before this case's next call.
                    checkpoint()
        with state_lock:
            applicable=[v['status'] for v in case['dimensions'].values() if v['status']!='not_applicable']
            case['status']='scored' if all(s=='scored' for s in applicable) else 'partial_error' if 'error' in applicable else 'not_scored'
            if not stop.is_set():counts['cases']+=1
            checkpoint()

    with state_lock:checkpoint()
    bar=tqdm(total=total_dimensions,desc=f'MLLM judge ({workers} workers)',unit='dim',
             dynamic_ncols=True,disable=not getattr(args,'progress',True),file=sys.stderr)
    def refresh_progress():
        with state_lock:
            bar.set_postfix(cases=f'{counts["cases"]}/{len(tasks)}',failed=counts['failed'],
                            cached=counts['cached'],api_calls=report['api_calls_made'],refresh=False)
            bar.update(counts['done']-bar.n)
        bar.refresh()
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='mllm-judge')
    def guarded_case(task):
        try:evaluate_case(task)
        except BaseException:
            stop.set();raise
    futures=set()
    try:
        futures={pool.submit(guarded_case,task) for task in tasks}
        last_refresh=0
        while futures:
            done,futures=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:future.result()
            if done or time.monotonic()-last_refresh>=1:
                refresh_progress();last_refresh=time.monotonic()
    finally:
        # Stop scheduling calls after interruption; in-flight responses finish
        # and are checkpointed before releasing the output lock.
        stop.set()
        pool.shutdown(wait=True,cancel_futures=True)
        refresh_progress();bar.close()
    errors=report['summary']['errors'];pending=report['summary']['pending']
    report['status']='error' if errors else 'not_scored' if pending else 'scored'
    checkpoint()
    return report,1 if errors else 2 if pending and args.require_scoring else 0


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATASET_ROOT)
    p.add_argument('--tasks',type=Path,help='Optional task JSONL; image paths remain relative to --dataset')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--predictions',type=Path,help='Directory of validated RGB native <id>.png exports')
    g.add_argument('--generation-manifest',type=Path,help='Verified raw/RGB export manifest with evaluation ROI')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids',help='Comma-separated case IDs; unknown IDs are errors')
    p.add_argument('--mode',choices=('no_reference','reference'),default='no_reference')
    p.add_argument('--config',type=Path,help='User-owned JSON config: base_url, model, api_key_env/api_key_file, timeout, max_retries')
    p.add_argument('--base-url');p.add_argument('--model');p.add_argument('--api-key-env')
    p.add_argument('--timeout',type=float,default=120)
    p.add_argument('--max-retries',type=int,default=2)
    p.add_argument('--workers',type=int,default=5,help='Concurrent cases / maximum simultaneous API requests (1-32; default 5)')
    p.add_argument('--progress',action=argparse.BooleanOptionalAction,default=True,help='Show live terminal progress (default enabled)')
    p.add_argument('--dry-run',action='store_true',help='Validate inputs and record not_scored; never send API requests')
    p.add_argument('--require-scoring',action='store_true',help='Exit2 when API/model missing or scoring intentionally pending')
    p.add_argument('--no-resume',action='store_false',dest='resume',help='Do not reuse matching completed dimension scores')
    return p


def main() -> int:
    args=parser().parse_args()
    try:
        report,code=evaluate(args)
        print(json.dumps({'status':report['status'],'summary':report['summary'],'output':str(args.output)},ensure_ascii=False))
        return code
    except Exception as exc:
        # Configuration/selection failures happen before API request construction.
        error=str(exc) if isinstance(exc,(JudgeError,FileNotFoundError,json.JSONDecodeError)) else type(exc).__name__
        error_path=args.output.with_name(args.output.name+'.error.json')
        error_report={'status':'error','error':error,'summary':{'expected':0,'scored':0},
                      'results':[],'evaluation_output_preserved':True}
        try:write_json(error_path,error_report)
        except OSError:pass
        print(json.dumps({'status':'error','error':error,'error_report':str(error_path),
                          'evaluation_output_preserved':True}),file=sys.stderr)
        return 1


if __name__=='__main__':sys.exit(main())
