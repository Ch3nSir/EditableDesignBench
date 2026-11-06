"""Separate text and asset MLLM profiles; no scoring is invented without an API.

Default: no-reference evaluation, using source + prediction + user instruction.
The original MIT-licensed IF/TA/VC/LP(/SE) prompts are snapshotted verbatim in
mllm_prompts.json; reference mode is explicit and separately identified.
Asset tasks declare their input protocol and use independent IF/OA/VC/LP
prompts with two full images, without masks, detail crops or GT.
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
import asset_judge_profile as asset_profile
import asset_response_validation as asset_validation
import text_judge_profile as text_profile

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


class ModelOutputError(JudgeError):
    """An asset model did not provide a usable output; not an API judge failure."""


def is_asset_task(task: dict) -> bool:
    return isinstance(task,dict) and task.get('input_protocol') == asset_profile.INPUT_PROTOCOL


def _task_profile(task: dict) -> str:
    if not isinstance(task,dict):raise JudgeError('Task entries must be objects')
    declared = task.get('input_protocol')
    if is_asset_task(task):return 'asset'
    if declared is not None and (not isinstance(declared,str) or not declared.startswith('text')):
        raise JudgeError('Unsupported task input_protocol: ' + str(declared))
    if declared is None and (task.get('asset_family_id') or str(task.get('id','')).startswith('asset_')):
        raise JudgeError('Asset tasks must explicitly declare ' + asset_profile.INPUT_PROTOCOL)
    return 'text'


def infer_profile(tasks: list[dict]) -> str:
    profiles={_task_profile(task) for task in tasks}
    if len(profiles)!=1:raise JudgeError('Evaluate text and asset tasks separately; mixed judge profiles are not supported')
    return next(iter(profiles))


def dimensions_for_task(task: dict, profile: str | None = None) -> tuple[str,...]:
    profile=profile or _task_profile(task)
    if profile=='asset':return asset_profile.DIMENSIONS
    knowledge=task.get('knowledge_prompt')
    return (*STANDARD_DIMENSIONS,'SE') if isinstance(knowledge,str) and knowledge.strip() else STANDARD_DIMENSIONS


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


def merge_cached_scores(cache: dict, records: list[dict], *, diagnostics: list | None = None,
                        source: str | None = None, section: str = 'records') -> None:
    """Keep only validated paid dimensions; unavailable inputs never erase them."""
    def rejected(error, **location):
        if diagnostics is None:raise JudgeError(error)
        diagnostics.append({'source':source,'section':section,**location,'error':error})
    if not isinstance(records,list):
        rejected('Invalid cached records list');return
    for index,record in enumerate(records):
        try:
            if not isinstance(record,dict):raise JudgeError('Invalid cached record')
            fingerprint=record.get('fingerprint')
            if not fingerprint:continue  # Pending/error placeholders have no scores.
            if not isinstance(record.get('id'),str) or not isinstance(fingerprint,str) or not re.fullmatch('[0-9a-f]{64}',fingerprint):
                raise JudgeError('Invalid cached case identity')
            dimensions=record.get('dimensions',{})
            if not isinstance(dimensions,dict):raise JudgeError('Invalid cached dimensions')
            profile=record.get('judge_profile','text')
            if profile not in ('text','asset'):raise JudgeError('Unknown cached judge profile')
        except JudgeError as exc:
            rejected(str(exc),record_index=index);continue
        scored={}
        for dimension,value in dimensions.items():
            try:
                if not isinstance(value,dict):raise JudgeError('Invalid cached dimension')
                if value.get('status')!='scored':continue
                allowed=asset_profile.DIMENSIONS if profile=='asset' else (*STANDARD_DIMENSIONS,'SE')
                if dimension not in allowed:raise JudgeError('Unknown cached score dimension')
                parsed=parse_score(json.dumps({dimension:value.get('score'),'rationale':value.get('rationale'),
                                               'evidence':value.get('evidence')}),dimension,asset_response=profile=='asset')
                scored[dimension]=dict(value,rationale=parsed['rationale'])
                if profile=='asset':
                    scored[dimension]['evidence']=parsed['evidence']
                if parsed.get('response_diagnostics'):
                    previous=value.get('response_diagnostics',[])
                    if not isinstance(previous,list):previous=[]
                    scored[dimension]['response_diagnostics']=list(dict.fromkeys(
                        [*(item for item in previous if isinstance(item,str)),*parsed['response_diagnostics']]))
            except JudgeError as exc:
                rejected(str(exc),record_index=index,dimension=dimension)
        if not scored:continue
        key=(record['id'],fingerprint)
        if key not in cache:cache[key]=dict(record,dimensions={})
        cache[key]['dimensions'].update(scored)


def load_score_cache(output: Path, *, diagnostics: list | None = None) -> dict:
    """Recover independently validated scores from each report/cache source."""
    cache={};issues=diagnostics if diagnostics is not None else []
    sidecar=cache_path(output)
    # The sidecar is committed first, so it may contain a newer paid response
    # than an interrupted report write. It must win duplicate dimensions.
    for path,is_sidecar in ((output,False),(sidecar,True)):
        if not path.exists():continue
        try:saved=json.loads(path.read_text())
        except (ValueError,UnicodeError):
            issues.append({'source':str(path),'error':'Invalid JSON in score cache source'});continue
        if not isinstance(saved,dict):
            issues.append({'source':str(path),'error':'Score cache source must be an object'});continue
        if is_sidecar and saved.get('schema_version')!=CACHE_SCHEMA_VERSION:
            issues.append({'source':str(path),'error':'Unsupported or invalid score cache schema'});continue
        sections=('records',) if is_sidecar else ('_resume_cache','results')
        for section in sections:
            records=saved.get(section,[] if section=='_resume_cache' else None)
            merge_cached_scores(cache,records,diagnostics=issues,source=str(path),section=section)
    if diagnostics is None and issues:
        print(f'MLLM cache recovery: ignored {len(issues)} invalid cache entries; original files remain unchanged.',file=sys.stderr)
    return cache


def preserve_cache_recovery_sources(diagnostics: list) -> None:
    """Keep damaged input bytes across later checkpoints and repeated resumes."""
    preserved={}
    for issue in diagnostics:
        source=issue['source']
        if source not in preserved:
            path=Path(source);content=path.read_bytes()
            target=path.with_name(path.name+'.rejected-'+digest(content)+'.bak')
            if not target.exists():write_bytes(target,content)
            preserved[source]=str(target)
        issue['preserved_file']=preserved[source]


def backup_existing(path: Path) -> None:
    if path.exists():write_bytes(path.with_name(path.name+'.bak'),path.read_bytes())


def load_tasks(path: Path, ids: str | None = None) -> list[dict]:
    tasks=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen=set()
    for t in tasks:
        if not isinstance(t,dict) or not isinstance(t.get('id'),str) or not isinstance(t.get('instruction'),str):
            raise JudgeError('Task entries require string id and instruction')
        if t['id'] in seen:raise JudgeError(f'Duplicate task ID: {t["id"]}')
        methods=asset_profile.METHODS if _task_profile(t)=='asset' else METHODS
        if t.get('operation') not in methods:raise JudgeError(f'Unsupported operation: {t["id"]}')
        seen.add(t['id'])
    if ids is not None:
        requested=[s.strip() for s in ids.split(',')]
        if any(not s for s in requested):raise JudgeError('Empty ID token in --ids')
        if len(requested)!=len(set(requested)):raise JudgeError('Duplicate requested ID in --ids')
        unknown=sorted(set(requested)-seen)
        if not requested or unknown:raise JudgeError('Unknown/empty --ids: '+','.join(unknown))
        tasks=[t for t in tasks if t['id'] in set(requested)]
    if not tasks:raise JudgeError('No tasks selected')
    infer_profile(tasks)
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


def current_profile_metadata(tasks: list[dict], mode: str = 'no_reference', text_prompt_profile: str = 'strict') -> dict:
    """Current judge contract without credentials; shared by artifact validators."""
    if infer_profile(tasks)=='asset':
        if mode!='no_reference':raise JudgeError('Asset judging does not accept a reference target')
        source=Path(asset_profile.__file__).resolve()
        return {'version':asset_profile.VERSION,'profile':'asset','mode':mode,
            'input_protocol':asset_profile.INPUT_PROTOCOL,
            'prompts_sha256':canonical_hash(asset_profile.PROMPTS),'prompt_source':str(source),
            'prompt_source_sha256':digest(source.read_bytes()),
            'user_message_template':'Original instruction, asset operation, full source and full prediction only',
            'message_template_revision':asset_profile.MESSAGE_TEMPLATE_REVISION,
            'criteria':list(asset_profile.DIMENSIONS),'score_range':[0,5],
            'temperature':0,'max_tokens_per_criterion':500,
            'separate_request_per_criterion':True,'independent_case_context':True,
            'images':'full source + full prediction native ROI; exactly two full images',
            'image_encoding':'Lossless RGB PNG; native full views only, no detail crops or resizing; files unchanged',
            'response_evidence_schema':dict(asset_profile.EVIDENCE_SCHEMA),
            'consistency_audit':'Conflicting target/operation claims flagged for review; scores are not adjusted',
            'view_version':asset_profile.VIEW_VERSION,'view_parameters':dict(asset_profile.VIEW_PARAMETERS),
            'operation_mapping':dict(asset_profile.METHODS),'reference_target_supplied':False,
            'editing_model_identity_supplied':False,'API_failure_scores':'null; never fabricated zero',
            'aggregate_means':'Conditional on valid outputs; null until all valid outputs judged, with full-selected coverage',
            'joint_qualification':{'dimensions':list(asset_profile.DIMENSIONS),'minimum_each':4,
                                   'denominator':'All selected cases, including model-output failures'},
            'API_connectivity_validated_by_dry_run':False}
    if text_prompt_profile not in ('strict','texteditbench'):
        raise JudgeError('Unknown text prompt profile')
    if text_prompt_profile=='strict':
        if mode not in ('no_reference','reference'):raise JudgeError('Unsupported text judge mode')
        source=Path(text_profile.__file__).resolve()
        return {'version':text_profile.VERSION,'profile':'text','text_profile':'strict','mode':mode,
            'prompts_sha256':canonical_hash(text_profile.PROMPTS),'prompt_source':str(source),
            'prompt_source_sha256':digest(source.read_bytes()),
            'message_template_revision':text_profile.MESSAGE_TEMPLATE_REVISION,
            'user_message_template':'Original instruction and official image/message ordering; strict text system prompts',
            'criteria':list(STANDARD_DIMENSIONS),'optional_SE':'Only explicit knowledge_prompt',
            'score_range':[0,5],'temperature':0,'max_tokens_per_criterion':500,
            'separate_request_per_criterion':True,'independent_case_context':True,
            'images':'source + prediction ROI' if mode=='no_reference' else 'source + prediction ROI + reference',
            'image_encoding':'Lossless RGB PNG; padding removed only from in-memory arrays; files unchanged',
            'operation_mapping':METHODS,'API_failure_scores':'null; never fabricated zero',
            'aggregate_means':'null unless every applicable expected case has a scored result',
            'API_connectivity_validated_by_dry_run':False}
    data=json.loads(PROMPT_FILE.read_text())
    if mode not in data['modes']:raise JudgeError('Unsupported text judge mode')
    source=resolve_prompt_source(data['modes'][mode])
    # Keep the text contract byte-for-byte equivalent at the canonical JSON
    # level: existing paid scores must remain reusable after this extension.
    return {'version':PROTOCOL_VERSION,'mode':mode,
        'prompts_sha256':digest(PROMPT_FILE.read_bytes()),'prompt_source':str(source),
        'prompt_source_sha256':data['modes'][mode]['source_sha256'],
        'user_message_template':'Verbatim official mode-specific text; validated RGB PNG image URLs',
        'criteria':list(STANDARD_DIMENSIONS),'optional_SE':'Only explicit knowledge_prompt',
        'score_range':[0,5],'temperature':0,'max_tokens_per_criterion':500,
        'separate_request_per_criterion':True,'independent_case_context':True,
        'images':'source + prediction ROI' if mode=='no_reference' else 'source + prediction ROI + reference',
        'image_encoding':'Lossless RGB PNG; padding removed only from in-memory arrays; files unchanged',
        'operation_mapping':METHODS,'API_failure_scores':'null; never fabricated zero',
        'aggregate_means':'null unless every applicable expected case has a scored result',
        'API_connectivity_validated_by_dry_run':False}


def validate_report_protocol(protocol: dict, tasks: list[dict]) -> bool:
    """Validate profile/prompt/view identity without reading API credentials."""
    if not isinstance(protocol,dict):return False
    try:
        expected=current_profile_metadata(tasks,protocol.get('mode','no_reference'),protocol.get('text_profile','texteditbench'))
        for key,value in expected.items():
            if key in ('temperature','max_tokens_per_criterion'):
                # Explicit alternate API parameters are validated below for
                # either track; prompt/view/rubric identity remains unchanged.
                if 'request_parameters' in protocol:continue
            if protocol.get(key)!=value:return False
        params=protocol.get('request_parameters',request_parameters({}))
        if not isinstance(params,dict):return False
        if 'max_completion_tokens' in params:
            if set(params)-{'max_completion_tokens','reasoning_effort','store'}:return False
            config={'request_profile':'reasoning_chat',**params}
            if request_parameters(config)!=params:return False
        elif (params.get('temperature')!=0 or params.get('max_tokens')!=500
              or set(params)-{'temperature','max_tokens','store'}):return False
        if 'store' in params and not isinstance(params['store'],bool):return False
        if protocol.get('temperature')!=params.get('temperature'):return False
        if protocol.get('max_tokens_per_criterion')!=params.get('max_tokens',params.get('max_completion_tokens')):return False
        return True
    except (JudgeError,ValueError,TypeError,KeyError,OSError):
        return False


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


def parse_score(content, dimension: str, *, asset_response: bool = False) -> dict:
    """Require an explicit numeric dimension; auxiliary omissions are diagnostic."""
    if isinstance(content,list):
        content=''.join(p['text'] for p in content if isinstance(p,dict)
                       and p.get('type') in ('text','output_text') and isinstance(p.get('text'),str))
    if isinstance(content,dict):data=content
    else:
        if not isinstance(content,str):raise JudgeError('Judge response content is not text or an object')
        text=content.strip()
        if text.startswith('```'):
            text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I)
            text=re.sub(r'\s*```$','',text)
        try:data=json.loads(text)
        except (ValueError,TypeError) as exc:raise JudgeError('Judge returned invalid JSON') from exc
    if not isinstance(data,dict) or dimension not in data:raise JudgeError('Judge JSON is missing score')
    value=data[dimension]
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=5 or not math.isfinite(value):
        raise JudgeError('Judge score must be numeric in [0,5]')
    rationale=data.get('rationale')
    diagnostics=[]
    if rationale is None:diagnostics.append('judge_rationale_missing')
    elif not isinstance(rationale,str):
        rationale=None;diagnostics.append('judge_rationale_invalid_type')
    elif not rationale.strip():diagnostics.append('judge_rationale_empty')
    result={dimension:value,'rationale':rationale}
    if asset_response:
        try:result['evidence']=asset_validation.validate_evidence(data,dimension)
        except ValueError as exc:raise JudgeError(str(exc)) from exc
        diagnostics.extend(asset_validation.inspect_evidence(data,dimension)[1])
    if diagnostics:result['response_diagnostics']=diagnostics
    return result


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
                  transport: Callable = post_json, sleeper: Callable = time.sleep, *, asset_response: bool = False) -> dict:
    retries=credentials['max_retries']
    if not 0<=retries<=10 or credentials['timeout']<=0:raise JudgeError('Invalid retry/timeout configuration')
    last='Judge request failed'
    for attempt in range(retries+1):
        try:
            data=transport(credentials['endpoint'],credentials['api_key'],
                {'model':credentials['model'],'messages':messages,
                 **credentials.get('request_parameters',request_parameters({}))},credentials['timeout'])
            choice=data['choices'][0]
            try:result=parse_score(choice['message']['content'],dimension,asset_response=asset_response)
            except JudgeError as exc:
                if choice.get('finish_reason')=='length':
                    raise JudgeError('Judge response hit token limit: '+str(exc)) from exc
                raise
            scored={'status':'scored','score':result[dimension],'rationale':result['rationale'],'attempts':attempt+1}
            if asset_response:scored['evidence']=result['evidence']
            diagnostics=list(result.get('response_diagnostics',[]))
            if choice.get('finish_reason')=='length':diagnostics.append('finish_reason_length_with_valid_score')
            if diagnostics:scored['response_diagnostics']=diagnostics
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
    profile=_task_profile(task)
    source_path=(args.dataset/task['image']).resolve()
    source=read_rgb(source_path)
    try:
        if args.generation_manifest is not None:
            from output_adapter import load_prediction
            record=manifest.get(task['id'])
            if record is None:
                if profile=='asset':raise ModelOutputError('Generation manifest is missing model output for case ID')
                raise JudgeError('Generation manifest is missing case ID')
            if profile=='asset' and record.get('status')!='complete':
                raise ModelOutputError('Generation did not produce a completed model output')
            prediction,provenance=load_prediction(task,args.dataset,record=record,manifest_path=args.generation_manifest)
        else:
            if args.predictions is None:raise JudgeError('Specify --predictions or --generation-manifest')
            from output_adapter import load_prediction
            prediction,validated=load_prediction(task,args.dataset,predictions_dir=args.predictions)
            # Keep the existing external-score identity while sharing the pixel
            # evaluator's RGB/RGBA channel and geometry policy.
            provenance={key:validated[key] for key in ('rgb_output','rgb_sha256','evaluation_roi')}
        if source.shape!=prediction.shape:raise JudgeError('Source and prediction evaluation ROI dimensions differ')
    except ModelOutputError:
        raise
    except (JudgeError,ValueError,OSError,KeyError) as exc:
        if profile=='asset':
            # A completed manifest is a provenance claim. Contradictory source,
            # instruction, ROI, adapter or file hashes must block publication,
            # rather than being swallowed as ordinary generation failures.
            if args.generation_manifest is not None and not isinstance(exc,OSError):
                raise JudgeError('Invalid generation provenance: '+str(exc)) from exc
            raise ModelOutputError('Unusable model output: '+str(exc)) from exc
        raise
    source_url=image_data_url(source);prediction_url=image_data_url(prediction)
    reference=None;reference_hash=None
    if profile=='asset' and args.mode!='no_reference':
        raise JudgeError('Asset judging does not accept a reference target')
    if args.mode=='reference':
        ref_path=args.dataset/'cases'/task['id']/'target.png'
        reference_array=read_rgb(ref_path)
        if reference_array.shape!=source.shape:raise JudgeError('Reference dimensions differ')
        reference=image_data_url(reference_array);reference_hash=digest(reference_array.tobytes())
    knowledge=task.get('knowledge_prompt')
    if knowledge is not None and (not isinstance(knowledge,str) or not knowledge.strip()):knowledge=None
    dimensions=dimensions_for_task(task,profile)
    methods=asset_profile.METHODS if profile=='asset' else METHODS
    view_metadata=None;view_urls=None
    if profile=='asset':
        try:
            views,view_metadata=asset_profile.full_views(source,prediction)
        except (ValueError,OSError,KeyError) as exc:
            raise JudgeError('Invalid asset judge views: '+str(exc)) from exc
        view_urls=[image_data_url(view) for view in views]
    # Preserve the paid-score identity when only validation diagnostics are
    # added to provenance. The image bytes/pixels, instruction and judge
    # protocol remain independently bound below and are revalidated first.
    legacy_fields=(('mode','manifest','raw_output','raw_sha256','rgb_output','rgb_sha256',
                    'source_size','canvas_size','evaluation_roi','padding','adapter_version')
                   if args.generation_manifest is not None else ('rgb_output','rgb_sha256','evaluation_roi'))
    identity_provenance={key:provenance[key] for key in legacy_fields if key in provenance}
    signature_payload={'source_sha256':digest(source_path.read_bytes()),
        'prediction_pixels_sha256':digest(prediction.tobytes()),'reference_pixels_sha256':reference_hash,
        'provenance':identity_provenance,'instruction':task['instruction'],'editing_method':methods[task['operation']],
        'knowledge_prompt':knowledge,'protocol_sha256':canonical_hash(protocol)}
    if profile=='asset':signature_payload.update(judge_profile='asset',input_protocol=task['input_protocol'],views=view_metadata)
    signature=canonical_hash(signature_payload)
    record={'id':task['id'],'source_id':task.get('source_id'),'domain':task.get('domain'),
        'operation':task['operation'],'instruction':task['instruction'],'editing_method':methods[task['operation']],
        'fingerprint':signature,'status':'not_scored','prediction_provenance':provenance,
        'dimensions':{d:{'status':'not_scored','score':None,'rationale':None} for d in dimensions}}
    if profile=='asset':record.update(judge_profile='asset',input_protocol=task['input_protocol'],generation_status='valid',judge_views=view_metadata)
    elif not knowledge:record['dimensions']['SE']={'status':'not_applicable','score':None,'reason':'No explicit knowledge_prompt; follows official optional SE rule.'}
    urls={'source':source_url,'prediction':prediction_url,'reference':reference}
    if profile=='asset':urls['views']=view_urls
    return record,urls


def summarize(records: list[dict], profile: str = 'text') -> dict:
    if profile=='asset':return summarize_assets(records)
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


def summarize_assets(records: list[dict]) -> dict:
    """Conditional quality means plus a fixed-denominator joint pass rate."""
    complete=all(row.get('status') in ('scored','model_failure') for row in records)
    def group(rows):
        expected=len(rows)
        valid=[row for row in rows if row.get('generation_status')=='valid']
        scored=[row for row in valid if row.get('status')=='scored']
        failures=sum(row.get('status')=='model_failure' for row in rows)
        ready=complete and len(scored)==len(valid)
        criteria={}
        for dimension in asset_profile.DIMENSIONS:
            values=[row['dimensions'][dimension]['score'] for row in valid
                    if row['dimensions'][dimension]['status']=='scored']
            criteria[dimension]={'mean':sum(values)/len(values) if ready and values else None,
                'scored':len(values),'expected':expected,'judge_expected':len(valid),
                'coverage':len(values)/expected if expected else None,
                'judge_coverage':len(values)/len(valid) if valid else None,
                'mean_denominator':'valid model outputs only'}
        means=[criteria[d]['mean'] for d in asset_profile.DIMENSIONS]
        qualified=sum(all(row['dimensions'][d]['score']>=4 for d in asset_profile.DIMENSIONS) for row in scored)
        return {'expected':expected,'valid_outputs':len(valid),'model_failures':failures,'scored':len(scored),
            'pending':sum(row.get('status')=='not_scored' for row in rows),
            'errors':sum(row.get('status') in ('error','partial_error') for row in rows),
            'formal_complete':ready,'valid_output_coverage':len(valid)/expected if expected else None,
            'criteria':criteria,'four_dimension_mean':sum(means)/4 if all(v is not None for v in means) else None,
            'joint_qualification':{'threshold_each':4,'qualified':qualified,'expected':expected,
                                   'rate':qualified/expected if ready and expected else None,
                                   'model_failures_count_as_unqualified':True}}
    result=group(records)
    result['consistency_review']={'cases':sum(bool(asset_validation.consistency_flags(row.get('dimensions',{}))) for row in records),
                                  'case_ids':[row['id'] for row in records if asset_validation.consistency_flags(row.get('dimensions',{}))],
                                  'scores_adjusted':False}
    result['by_operation']={operation:group([row for row in records if row['operation']==operation])
                            for operation in asset_profile.METHODS}
    return result


def unprocessed_record(task: dict) -> dict:
    if _task_profile(task)=='asset':
        return {'id':task['id'],'operation':task['operation'],'judge_profile':'asset',
                'generation_status':'pending','status':'not_scored',
                'dimensions':{d:{'status':'not_scored','score':None,'rationale':None,'reason':'not_processed'}
                              for d in asset_profile.DIMENSIONS}}
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
    profile=infer_profile(tasks)
    text_prompt_profile=getattr(args,'text_profile','strict')
    metadata=current_profile_metadata(tasks,args.mode,text_prompt_profile)
    prompts=(asset_profile.PROMPTS if profile=='asset' else text_profile.PROMPTS if text_prompt_profile=='strict'
             else json.loads(PROMPT_FILE.read_text())['modes'][args.mode]['prompts'])
    credentials=load_credentials(args)
    protocol={**metadata,'judge_model':credentials['model'],'endpoint':credentials['endpoint']}
    parameters=credentials['request_parameters']
    if parameters!=request_parameters({}):
        # Leave old official fingerprints identical. Explicit alternate API
        # parameters participate in input/cache and cross-model comparison IDs.
        protocol.update(request_parameters=parameters,temperature=parameters.get('temperature'),
                        max_tokens_per_criterion=parameters.get('max_tokens',parameters.get('max_completion_tokens')))
    cache_diagnostics=[]
    old=load_score_cache(args.output,diagnostics=cache_diagnostics)
    manifest=load_manifest(args.generation_manifest)
    # Validate everything that can fail before replacing the current report.
    # One recoverable snapshot per run avoids a backup per dimension/request.
    preserve_cache_recovery_sources(cache_diagnostics)
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
    if cache_diagnostics:
        report['storage']['cache_recovery']=cache_diagnostics
        print(f'MLLM cache recovery: ignored {len(cache_diagnostics)} invalid cache entries; damaged sources preserved in .rejected-*.bak files.',file=sys.stderr,flush=True)
    positions={r['id']:i for i,r in enumerate(report['results'])}
    state_lock=Lock();stop=Event()
    counts={'done':0,'failed':0,'cached':0,'cases':0,'model_failures':0}
    report['resume']={'enabled':args.resume,'cached_dimensions':0}
    if args.resume and old and getattr(args,'progress',True):
        available=sum(len(record['dimensions']) for record in old.values())
        print(f'MLLM resume: {available} successful cached dimensions found; validating inputs before reuse.',file=sys.stderr,flush=True)
    total_dimensions=sum(len(dimensions_for_task(task,profile)) for task in tasks)
    report['execution']={'workers':workers,'concurrency_unit':'case','maximum_simultaneous_api_requests':workers}
    if profile=='asset':
        validation_source=Path(asset_validation.__file__).resolve()
        # Local acceptance/audit policy is separate from the API request identity.
        # Changing it does not require paying again for already-valid responses.
        report['response_validation']={'version':asset_validation.VERSION,
            'source':str(validation_source),'source_sha256':digest(validation_source.read_bytes()),
            'semantic_disagreements':'review_only','scores_modified':False,
            'retry_on_semantic_disagreement':False,'auxiliary_evidence':'optional; usable supplied fields normalized with diagnostics',
            'retry_on_auxiliary_evidence_omission':False}
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
        report['summary']=summarize(report['results'],profile);write_json(args.output,report)
    def safe_error(exc):
        text=str(exc) if isinstance(exc,(JudgeError,FileNotFoundError,ValueError)) else type(exc).__name__
        return text.replace(credentials['api_key'],'[REDACTED]') if credentials['api_key'] else text

    def failed_case(task,exc):
        with state_lock:
            existing=report['results'][positions[task['id']]]
            model_failure=profile=='asset' and isinstance(exc,ModelOutputError)
            existing.update(status='model_failure' if model_failure else 'error',error=safe_error(exc))
            if profile=='asset':
                existing.update(generation_status='failed' if model_failure else existing.get('generation_status','unknown'),
                                failure_type='model_output' if model_failure else 'evaluation_input')
                if model_failure:
                    existing['joint_qualified']=False;counts['model_failures']+=1
            for value in existing['dimensions'].values():
                if value['status']=='not_scored':
                    value.update(status='not_applicable' if model_failure else 'error',error=safe_error(exc))
                    counts['done']+=1;counts['failed']+=not model_failure
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
                        parse_score(json.dumps({d:cached.get('score'),'rationale':cached.get('rationale'),
                                              'evidence':cached.get('evidence')}),d,asset_response=profile=='asset')
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
                    messages=(asset_profile.build_messages(dimension,task,urls['views']) if profile=='asset' else
                              build_messages(dimension,prompts[dimension],task,urls['source'],urls['prediction'],urls['reference']))
                    result=request_score(dimension,messages,credentials,counted_transport,sleeper,asset_response=True) if profile=='asset' else request_score(dimension,messages,credentials,counted_transport,sleeper)
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
            if profile=='asset':
                case['consistency_flags']=asset_validation.consistency_flags(case['dimensions'])
                case['joint_qualified']=(all(case['dimensions'][d]['score']>=4 for d in asset_profile.DIMENSIONS)
                                         if case['status']=='scored' else None)
            if not stop.is_set():counts['cases']+=1
            checkpoint()

    with state_lock:checkpoint()
    bar=tqdm(total=total_dimensions,desc=f'MLLM judge ({workers} workers)',unit='dim',
             dynamic_ncols=True,disable=not getattr(args,'progress',True),file=sys.stderr)
    def refresh_progress():
        with state_lock:
            bar.set_postfix(cases=f'{counts["cases"]}/{len(tasks)}',failed=counts['failed'],
                            cached=counts['cached'],api_calls=report['api_calls_made'],
                            **({'model_failed':counts['model_failures']} if profile=='asset' else {}),refresh=False)
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
    if profile=='asset' and report['status']=='scored' and report['summary']['model_failures']:
        report['status']='scored_with_model_failures'
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
    p.add_argument('--text-profile',choices=('strict','texteditbench'),default='strict',help='Text judge rubric; ignored by the asset track')
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
