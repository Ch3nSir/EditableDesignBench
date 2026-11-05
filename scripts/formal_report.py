"""Validate one model's formal evaluation without contacting a judge API.

Every run writes its own report. Dataset summary.json is never modified.
"""
from __future__ import annotations

from settings import DATASET_ROOT, MODEL_ROOT

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from evaluation_binding import canonical_hash, dataset_identity, load_tasks
from mllm_evaluate import (STANDARD_DIMENSIONS, infer_profile, parse_score,
                           summarize as judge_summary, validate_report_protocol)
from asset_judge_profile import DIMENSIONS as ASSET_DIMENSIONS
from output_adapter import load_generation_records
from smoke_report import _generation_provenances, _judge_inputs_match, pixel_inputs_match

ROOT=Path(__file__).resolve().parents[1]
REPORT_VERSION='texteditbench-formal-report-v1'
ASSET_REPORT_VERSION='editabledesignbench-assets-formal-report-v1'
PIXEL_NAMES=('mse','psnr','ssim','lpips')
PINNED_GENERATIONS={'qwen-image21-local-generation-v2','multi-image-edit-generation-v1'}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name)
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        temp.replace(path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)


def _read(path,errors,label,required=True):
    try:
        if not path.exists():
            if required:errors.append(label+': missing '+path.name)
            return None
        value=json.loads(path.read_text())
        if not isinstance(value,dict):raise ValueError('report must be an object')
        return value
    except (OSError,ValueError) as exc:
        errors.append(label+': '+str(exc));return None


def _number(value,name):
    if isinstance(value,bool):raise ValueError(name+': boolean score')
    if value=='inf' and name=='psnr':return math.inf
    if not isinstance(value,(int,float)):raise ValueError(name+': nonnumeric score')
    value=float(value)
    if not math.isfinite(value) and not (name=='psnr' and value==math.inf):
        raise ValueError(name+': nonfinite score')
    if name in ('mse','lpips','psnr') and value<0:raise ValueError(name+': negative score')
    if name=='mse' and value>65025:raise ValueError('MSE outside uint8 range')
    if name=='ssim' and not -1<=value<=1:raise ValueError('SSIM outside range')
    return value


def _pixel_axes(pixel,expected):
    """Independently validate statuses, full denominators and each mean."""
    rows=pixel['results'];summary=pixel.get('summary') or {}
    means=summary.get('metrics');coverage=summary.get('metric_coverage')
    names=set(PIXEL_NAMES)
    if not isinstance(means,dict) or set(means)!={'avg_'+name for name in names}:
        raise ValueError('Pixel summary metric keys differ')
    if not isinstance(coverage,dict) or set(coverage)!=names:raise ValueError('Pixel coverage keys differ')
    for row in rows:
        states=row.get('metric_status');failures=row.get('metric_errors')
        if not isinstance(states,dict) or set(states)!=names or any(s not in ('ok','error','pending') for s in states.values()):
            raise ValueError('Pixel metric states differ')
        if not isinstance(failures,dict) or set(failures)-names:raise ValueError('Pixel metric error keys differ')
        good=sum(s=='ok' for s in states.values())
        row_status=('ok' if good==len(names) else 'pending' if all(s=='pending' for s in states.values())
                    else 'partial' if good else 'error')
        if row.get('status')!=row_status:raise ValueError('Pixel row status differs from metric states')
    valid=sum(row['status']=='ok' for row in rows);pending=sum(row['status']=='pending' for row in rows)
    counts={'expected':len(expected),'valid':valid,'failed':len(expected)-valid-pending,'pending':pending,
            'partial':sum(row['status']=='partial' for row in rows),
            'coverage':valid/len(expected),'complete':valid==len(expected)}
    for key,value in counts.items():
        if summary.get(key)!=value or pixel.get(key)!=value:raise ValueError('Pixel '+key+' count differs')
    values=dict.fromkeys(PIXEL_NAMES);validated_coverage={};errors=[];axis_complete={}
    for name in PIXEL_NAMES:
        try:
            states=[row['metric_status'][name] for row in rows]
            good=states.count('ok');waiting=states.count('pending')
            expected_coverage={'expected':len(expected),'valid':good,'failed':len(expected)-good-waiting,
                'pending':waiting,'coverage':good/len(expected),'complete':good==len(expected)}
            if coverage[name]!=expected_coverage:raise ValueError('coverage differs from per-case states')
            numbers=[]
            for row in rows:
                status=row['metric_status'][name];error=row['metric_errors'].get(name)
                if status=='ok':
                    if error is not None:raise ValueError('successful metric has an error')
                    number=_number(row.get(name),name)
                    if name in ('mse','psnr') and number!=round(number,4):raise ValueError('official per-case rounding differs')
                    numbers.append(number)
                else:
                    if row.get(name) is not None:raise ValueError('failed/pending metric needs a null value')
                    if status=='error' and (not isinstance(error,str) or not error):raise ValueError('failed metric needs its reason')
                    if status=='pending' and error is not None:raise ValueError('pending metric has an error')
            aggregate=means['avg_'+name]
            if good==len(expected):
                calculated=float(np.mean(numbers));actual=_number(aggregate,name)
                if actual!=calculated:raise ValueError('mean differs from per-case arithmetic mean')
                values[name]=aggregate;axis_complete[name]=True
            else:
                if aggregate is not None:raise ValueError('incomplete metric mean must be null')
                axis_complete[name]=False;errors.append(name+': incomplete metric coverage')
            validated_coverage[name]=expected_coverage
        except Exception as exc:
            errors.append(name+': '+str(exc));axis_complete[name]=False
    return values,validated_coverage,bool(all(axis_complete.values())),errors


def _generation_inputs(gen,protocol,expected,dataset,manifest):
    if protocol.get('generation_protocol_version') in PINNED_GENERATIONS:
        content=protocol.get('content_provenance') or {}
        fingerprint=content.get('sha256')
        if not fingerprint or canonical_hash({k:v for k,v in content.items() if k!='sha256'})!=fingerprint:
            raise ValueError('Generation content inventory mismatch')
        if gen.get('content_provenance_sha256')!=fingerprint or gen.get('generation_protocol_version')!=protocol['generation_protocol_version']:
            raise ValueError('Generation header content identity mismatch')
        for record in gen.get('records',[]):
            if record.get('content_provenance_sha256')!=fingerprint:raise ValueError('Generation record content identity mismatch')
    if protocol.get('generation_protocol_version')=='multi-image-edit-generation-v1':
        for key in ('model_id','run_id'):
            if not isinstance(protocol.get(key),str) or not protocol[key].strip():raise ValueError('Generation '+key+' required')
            if gen.get(key)!=protocol[key]:raise ValueError('Generation header '+key+' mismatch')
            if any(r.get(key)!=protocol[key] for r in gen.get('records',[])):
                raise ValueError('Generation record '+key+' mismatch')
    return _generation_provenances(gen,protocol,expected,dataset,manifest)


def _asset_generation_header(gen,protocol,expected):
    """Pin an incomplete asset run without pretending its outputs all exist.

    Individual successful/missing predictions are revalidated by the judge
    binding helper. This only permits retaining a genuinely fixed-denominator
    semantic result when generation or pixel coverage is incomplete.
    """
    if not isinstance(gen,dict) or protocol.get('generation_protocol_version') not in PINNED_GENERATIONS:
        raise ValueError('Asset semantic report requires a pinned generation manifest')
    fingerprint=protocol.get('protocol_sha256')
    if (canonical_hash({k:v for k,v in protocol.items() if k!='protocol_sha256'})!=fingerprint
            or gen.get('protocol_sha256')!=fingerprint or gen.get('expected_ids')!=expected):
        raise ValueError('Asset generation protocol/selection differs')
    content=protocol.get('content_provenance') or {};content_sha=content.get('sha256')
    if (not content_sha or canonical_hash({k:v for k,v in content.items() if k!='sha256'})!=content_sha
            or gen.get('content_provenance_sha256')!=content_sha
            or gen.get('generation_protocol_version')!=protocol['generation_protocol_version']):
        raise ValueError('Asset generation content identity differs')
    records=gen.get('records')
    if not isinstance(records,list):raise ValueError('Asset generation records missing')
    ids=[r.get('id') for r in records]
    if len(ids)!=len(set(ids)) or set(ids)-set(expected):raise ValueError('Asset generation record IDs differ')
    for name in ('model_id','run_id'):
        if (not isinstance(protocol.get(name),str) or not protocol[name].strip()
                or gen.get(name)!=protocol[name]):raise ValueError('Asset generation header '+name+' differs')
    for record in records:
        for name,value in (('protocol_sha256',fingerprint),('content_provenance_sha256',content_sha),
                           ('model_id',protocol['model_id']),('run_id',protocol['run_id'])):
            if record.get(name)!=value:raise ValueError('Asset generation record '+name+' differs')
    return True


def _judge_values(judge,expected,tasks):
    selected=[tasks[cid] for cid in expected]
    profile=infer_profile(selected)
    names=ASSET_DIMENSIONS if profile=='asset' else (*STANDARD_DIMENSIONS,'SE')
    if profile=='asset' and not validate_report_protocol(judge.get('protocol') or {},selected):
        raise ValueError('Asset judge profile/prompt/view protocol differs')
    rows=judge.get('results')
    if not isinstance(rows,list) or len(rows)!=len(expected) or {r.get('id') for r in rows}!=set(expected):
        raise ValueError('MLLM requested case set differs')
    for row in rows:
        task=tasks[row['id']];dimensions=row.get('dimensions')
        if not isinstance(dimensions,dict) or set(dimensions)!=set(names):
            raise ValueError('MLLM dimensions differ')
        if profile=='asset' and row.get('status')=='model_failure':
            if (row.get('generation_status')!='failed'
                    or any(v.get('status')!='not_applicable' or v.get('score') is not None
                           for v in dimensions.values())):
                raise ValueError('Invalid model-output-failure score')
            continue
        knowledge=task.get('knowledge_prompt')
        has_se=isinstance(knowledge,str) and bool(knowledge.strip())
        applicable=[]
        for dimension,value in dimensions.items():
            status=value.get('status')
            if dimension=='SE' and not has_se:
                if status!='not_applicable' or value.get('score') is not None:raise ValueError('Inapplicable SE has a score')
                continue
            applicable.append(status)
            if status=='scored':
                parse_score(json.dumps({dimension:value.get('score'),'rationale':value.get('rationale')}),dimension)
            elif status not in ('not_scored','error') or value.get('score') is not None:
                raise ValueError('Invalid pending/error MLLM score')
        if all(s=='scored' for s in applicable) and row.get('status')!='scored':raise ValueError('MLLM scored row status inconsistent')
        if row.get('status')=='scored' and not all(s=='scored' for s in applicable):raise ValueError('Incomplete MLLM row claims scored')
    recalculated=judge_summary(rows,profile=profile)
    if judge.get('summary')!=recalculated:raise ValueError('MLLM summary differs from requested case scores')
    if profile=='asset':
        complete=(judge.get('status') in ('scored','scored_with_model_failures')
                  and recalculated['formal_complete'] and recalculated['pending']==0 and recalculated['errors']==0)
    else:
        complete=(judge.get('status')=='scored' and recalculated['scored']==len(expected)
                  and recalculated['pending']==0 and recalculated['errors']==0)
    return recalculated,complete


def _pixel_comparison_protocol(pixel):
    if not pixel:return None
    protocol=pixel.get('protocol') or {}
    return {key:protocol.get(key) for key in ('version','artifact_binding_version','software_versions','device','cpu_threads')}|{
        'official_sources':{k:v.get('sha256') for k,v in protocol.get('sources',{}).items()},
        'evaluator_sources':{k:v.get('sha256') for k,v in protocol.get('evaluator_sources',{}).items()}}


def _judge_comparison_protocol(judge):
    if not judge:return None
    protocol=judge.get('protocol') or {}
    value={key:protocol.get(key) for key in (
        'version','mode','judge_model','endpoint','prompts_sha256','prompt_source_sha256','criteria',
        'score_range','temperature','max_tokens_per_criterion','separate_request_per_criterion',
        'independent_case_context','images','image_encoding','operation_mapping','optional_SE',
        'profile','input_protocol','reference_target_supplied','editing_model_identity_supplied',
        'view_version','view_parameters','joint_qualification')}
    if 'request_parameters' in protocol:value['request_parameters']=protocol['request_parameters']
    for key in ('text_profile','message_template_revision'):
        if key in protocol:value[key]=protocol[key]
    return value


def expected_case_ids(protocol,tasks,requested=None):
    expected=protocol.get('case_ids')
    if not isinstance(expected,list) or not expected or any(not isinstance(x,str) for x in expected) or len(set(expected))!=len(expected):
        raise ValueError('Generation protocol requires unique case_ids')
    if set(expected)-set(tasks):raise ValueError('Generation case IDs not in dataset')
    if requested is not None:
        selected=list(tasks) if requested=='all' else [x.strip() for x in requested.split(',')]
        if not selected or any(not x for x in selected) or len(selected)!=len(set(selected)) or set(selected)!=set(expected):
            raise ValueError('Requested evaluation case set differs from generation protocol')
    return expected


def evaluate_run(run_dir,dataset_root=ROOT,*,mode='formal',expected_ids=None):
    """Return (report, exit code); no mutable dataset or run file is touched."""
    if mode not in ('formal','pixel-only','dry-run'):raise ValueError('Unknown report mode')
    run=Path(run_dir).resolve();dataset=Path(dataset_root).resolve();errors=[]
    protocol=_read(run/'protocol.json',errors,'generation protocol') or {}
    generation=_read(run/'generation_manifest.json',errors,'generation')
    pixel=_read(run/'pixel_scores.json',errors,'pixel')
    judge_name='mllm_pending.json' if mode=='dry-run' else 'mllm_scores.json'
    judge=_read(run/judge_name,errors,'MLLM',required=mode!='pixel-only') if mode!='pixel-only' else None
    tasks=load_tasks(dataset);identity=dataset_identity(dataset)
    expected=expected_case_ids(protocol,tasks,expected_ids)
    profile=infer_profile([tasks[cid] for cid in expected])
    dimensions=ASSET_DIMENSIONS if profile=='asset' else STANDARD_DIMENSIONS
    semantic_names=dimensions if profile=='asset' else (*dimensions,'SE')
    manifest=run/'generation_manifest.json';provenances={};generation_ok=False
    try:
        if generation is None:raise ValueError('No generation manifest')
        provenances=_generation_inputs(generation,protocol,expected,dataset,manifest);generation_ok=True
    except Exception as exc:errors.append('generation: '+str(exc))
    pixel_errors=[];pixel_values=dict.fromkeys(PIXEL_NAMES);pixel_coverage={};pixel_ok=False
    pixel_inputs_ok=generation_ok and pixel_inputs_match(pixel,expected,dataset_root=dataset,
        manifest_path=manifest,provenances=provenances,errors=pixel_errors)
    errors.extend('pixel: '+e for e in pixel_errors)
    if pixel_inputs_ok:
        try:
            pixel_values,pixel_coverage,pixel_ok,axis_errors=_pixel_axes(pixel,expected)
            errors.extend('pixel: '+error for error in axis_errors)
        except Exception as exc:errors.append('pixel: '+str(exc))
    elif not pixel_errors:errors.append('pixel: generation or pixel report incomplete')
    judge_ok=False;judge_inputs_ok=False;semantic_summary=None
    semantic_generation_header_valid=generation_ok
    criteria={d:{'mean':None,'scored':0,'expected':len(expected),'coverage':0.0} for d in dimensions}
    if profile=='text':
        criteria['SE']={'mean':None,'scored':0,'expected':sum(bool(t.get('knowledge_prompt')) for cid,t in tasks.items() if cid in expected),'coverage':None}
    if judge is not None:
        try:
            if not generation_ok:
                if profile!='asset':raise ValueError('Generation input validation failed')
                semantic_generation_header_valid=_asset_generation_header(generation,protocol,expected)
            _judge_inputs_match(judge,expected,dataset,manifest,provenances)
            recalculated,judge_ok=_judge_values(judge,expected,tasks)
            criteria=recalculated['criteria'];judge_inputs_ok=True;semantic_summary=recalculated
            if mode=='dry-run' and judge.get('api_calls_made')!=0:raise ValueError('Dry-run report contains API calls')
            if any(value['scored'] for value in criteria.values()) and (not isinstance(judge['protocol'].get('judge_model'),str) or not judge['protocol']['judge_model'].strip()):
                raise ValueError('Scored report has no judge model')
        except Exception as exc:
            errors.append('MLLM: '+str(exc));judge_ok=False;judge_inputs_ok=False
    semantic_complete=bool(judge_ok and mode!='dry-run')
    if mode=='formal' and not semantic_complete:errors.append('MLLM: semantic scoring incomplete')
    requested_complete=generation_ok and pixel_ok and (semantic_complete if mode=='formal' else judge_inputs_ok if mode=='dry-run' else True)
    semantic_values={name:(criteria[name]['mean'] if judge_inputs_ok and mode!='dry-run' else None) for name in semantic_names}
    if profile=='asset':
        semantic_values['joint_qualification_rate']=(semantic_summary['joint_qualification']['rate']
            if judge_inputs_ok and mode!='dry-run' else None)
    binding_rows=[]
    if pixel:
        for row in pixel.get('results',[]):
            value=row.get('input_binding') or {}
            binding_rows.append({key:value.get(key) for key in ('id','task_sha256','source_sha256','instruction_sha256','masks')})
    model_id=protocol.get('model_id') or Path(protocol.get('model_path') or 'unknown-model').name
    run_id=protocol.get('run_id') or protocol.get('protocol_sha256','unknown-run')[:16]
    report={'version':ASSET_REPORT_VERSION if profile=='asset' else REPORT_VERSION,
        'track':profile,'semantic_dimensions':list(dimensions),
        'mode':mode,'model_id':model_id,'run_id':run_id,'run_dir':str(run),
        'created_at':datetime.now(timezone.utc).isoformat(),'status':'complete' if requested_complete and mode!='dry-run' else 'validated_dry_run' if requested_complete else 'incomplete',
        'complete':bool(requested_complete and mode!='dry-run'),'requested_mode_valid':bool(requested_complete),
        'generation_complete':generation_ok,'pixel_inputs_valid':pixel_inputs_ok,
        **({'semantic_generation_header_valid':semantic_generation_header_valid} if profile=='asset' else {}),
        'pixel_complete':pixel_ok,'semantic_inputs_valid':judge_inputs_ok,'semantic_complete':semantic_complete,
        'full_evaluation_complete':generation_ok and pixel_ok and semantic_complete,
        'expected_cases':len(expected),'case_ids':expected,'scope':'all' if set(expected)==set(tasks) else 'subset',
        'dataset':identity,'resolution_protocol':protocol.get('resolution_protocol','legacy-unspecified'),
        'generation_protocol_sha256':protocol.get('protocol_sha256'),
        'generation_identity_version':protocol.get('generation_protocol_version','legacy-unpinned'),
        'metrics':{**pixel_values,**semantic_values},
        'pixel_coverage':pixel_coverage,
        'semantic_coverage':{key:{k:v for k,v in value.items() if k!='mean'} for key,value in criteria.items()},
        'mllm_api_calls_in_last_stage':(judge or {}).get('api_calls_made',0),
        'comparison_protocol':{'track':profile,'dataset':{k:v for k,v in identity.items() if k!='dataset_root'},
            'case_ids':sorted(expected),'resolution_protocol':protocol.get('resolution_protocol','legacy-unspecified'),
            'evaluation_data_sha256':canonical_hash(sorted(binding_rows,key=lambda row:str(row.get('id'))))
                if len(binding_rows)==len(expected) and all(row.get('id') and row.get('masks') for row in binding_rows) else None,
            'pixel':_pixel_comparison_protocol(pixel),'judge':_judge_comparison_protocol(judge)},
        'validation_errors':errors,'note':'Each metric keeps its own validated full-case mean; incomplete axes are null. Overall completeness requires all applicable axes. Historical caches do not enter the requested denominator.'}
    if profile=='asset':
        report['semantic_summary']=semantic_summary
        report['joint_qualification']=(semantic_summary or {}).get('joint_qualification')
        report['by_operation']=(semantic_summary or {}).get('by_operation',{})
        report['note']='Asset quality means are conditional on valid outputs with coverage. Joint qualification uses all selected cases; model-output failures are unqualified. API/input errors keep formal means and qualification rate null. Text and asset scores are separate protocols.'
    return report,0 if requested_complete else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--pixel-only',action='store_true');modes.add_argument('--require-semantic',action='store_true')
    modes.add_argument('--dry-run',action='store_true')
    parser.add_argument('--expected-ids',help='all or comma-separated IDs; must match the generation protocol')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);output=args.output or args.run_dir/'report.json'
    try:
        report,code=evaluate_run(args.run_dir,args.dataset_root,mode='pixel-only' if args.pixel_only else 'dry-run' if args.dry_run else 'formal',expected_ids=args.expected_ids)
    except Exception as exc:
        report={'version':REPORT_VERSION,'status':'invalid_request','complete':False,'full_evaluation_complete':False,'validation_errors':[str(exc)]};code=2
    write_json(output,report)
    print(json.dumps({'status':report['status'],'complete':report['complete'],'output':str(output)},ensure_ascii=False))
    return code


if __name__=='__main__':raise SystemExit(main())
