"""Local TextEditBench official MSE/PSNR/SSIM/LPIPS with verified output ROI.

Metric math and successful-case means reuse the unmodified official source.
Missing/invalid outputs still fail explicitly; model output files stay unchanged.
"""
from settings import DATASET_ROOT, MODEL_ROOT
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import importlib.metadata
import json
import math
import shutil
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from output_adapter import load_generation_records, load_prediction
from evaluation_binding import (BINDING_VERSION, ASSET_PIXEL_PROTOCOL, binding_version, dataset_track,
    preservation_mask_path, canonical_hash, case_binding, dataset_identity, pixel_sources, write_json)
from official_pixel_metrics import (
    METRICS, PROTOCOL_VERSION, OFFICIAL_EVALUATION, protocol_info, score_case,
)

ROOT=Path(__file__).resolve().parents[1]

# Verified prior wrappers: metric formulas and accepted RGB/ROI inputs remain
# identical. Removing the alpha gate admits new inputs without changing scores
# of previously accepted inputs. Unknown implementation changes invalidate reuse.
COMPATIBLE_WRAPPERS={
    'evaluate.py':{'05e968d463b0583e3a1d251007190677c2b8ead709088ee4cde5af91d33dac13',
                   '3ec5a54f267b25c4363ab375790446356108805e8f52b61febfe5069d1fbc0f1',
                   '38aaf28fbafa36085a58e54710f9b4d838b2f72e8e15b12c5c0f2eaaff726817'},
    'official_pixel_metrics.py':{'074806c9dd10dbbd26c8375cb632b9230aecdd10fe1af8e21d537fd6f8fa6055'},
    'output_adapter.py':{'0d22becf61d07def54a37a4f5615b86cbc91de757ee571c359360021a3fe0169',
                         '85d656ea28734923af930de286a768784ac925a01fa1ad0c9615bb06d7683049'},
}


def cache_path(output):return output.with_name(output.name+'.cache.json')


@contextmanager
def output_lock(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_name(output.name+'.lock').open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Pixel output already has an active evaluation') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def _protocol_key(protocol):
    return canonical_hash({k:v for k,v in protocol.items() if k!='runtime_warnings'})


def _compatible_protocol(previous,current):
    ignored={'runtime_warnings','evaluator_sources'}
    if {k:v for k,v in previous.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}:return False
    old_sources=previous.get('evaluator_sources',{});new_sources=current['evaluator_sources']
    if not isinstance(old_sources,dict):return False
    if set(old_sources)!=set(new_sources):return False
    for name,source in new_sources.items():
        old=old_sources[name]
        if not isinstance(old,dict):return False
        if old.get('path')!=source['path']:return False
        if not isinstance(old.get('sha256'),str):return False
        if old.get('sha256') not in {source['sha256'],*COMPATIBLE_WRAPPERS.get(name,set())}:return False
    return True


def _cached_number(name,value):
    if name=='psnr' and value=='inf':return math.inf
    if isinstance(value,bool) or not isinstance(value,(int,float)):return None
    try:value=float(value)
    except OverflowError:return None
    if not math.isfinite(value) and not (name=='psnr' and value==math.inf):return None
    if name in ('mse','psnr','lpips') and value<0:return None
    if name=='mse' and value>65025:return None
    if name=='ssim' and not -1<=value<=1:return None
    return value


def merge_pixel_cache(cache,protocol,rows,*,source=None):
    key=_protocol_key(protocol)
    for index,row in enumerate(rows):
        try:
            if not isinstance(row,dict):raise ValueError('record must be an object')
            states=row.get('metric_status',{})
            if not isinstance(states,dict):raise ValueError('metric_status must be an object')
            good={}
            for name in METRICS:
                if states.get(name)!='ok':continue
                value=_cached_number(name,row.get(name))
                if value is not None:good[name]=value
                elif source:cache['diagnostics'].append(f'{source} record {index}: invalid successful {name}; ignored')
            if not good:continue
            binding=row.get('input_binding');provenance=row.get('provenance')
            if not isinstance(row.get('id'),str) or not row['id']:raise ValueError('invalid record ID')
            if not isinstance(binding,dict) or not isinstance(binding.get('sha256'),str):raise ValueError('missing input binding')
            if not isinstance(provenance,dict):raise ValueError('missing provenance')
            if canonical_hash({k:v for k,v in binding.items() if k!='sha256'})!=binding['sha256']:
                raise ValueError('input binding fingerprint mismatch')
            if binding.get('id')!=row['id']:raise ValueError('input binding ID mismatch')
            cache['protocols'][key]=deepcopy(protocol)
            slot=(key,row['id'],binding['sha256'])
            if slot not in cache['records']:
                cache['records'][slot]={'protocol_key':key,'id':row['id'],'input_binding':deepcopy(binding),
                                        'provenance':deepcopy(provenance),'scores':{}}
            cache['records'][slot]['scores'].update(good)
        except (TypeError,ValueError,KeyError) as exc:
            if source:cache['diagnostics'].append(f'{source} record {index}: {exc}; ignored')


def load_pixel_cache(output):
    cache={'protocols':{},'records':{},'diagnostics':[]}
    if output.exists():
        try:
            report=json.loads(output.read_text())
            if (not isinstance(report,dict) or not isinstance(report.get('results'),list)
                    or not isinstance(report.get('protocol'),dict)):
                raise ValueError('Invalid existing pixel report')
            merge_pixel_cache(cache,report['protocol'],report['results'],source=output.name)
        except (OSError,ValueError,TypeError) as exc:
            cache['diagnostics'].append(f'{output.name}: {exc}; report cache ignored')
    path=cache_path(output)
    if path.exists():
        try:
            saved=json.loads(path.read_text())
            if (not isinstance(saved,dict) or saved.get('schema_version')!=1
                    or not isinstance(saved.get('protocols'),dict) or not isinstance(saved.get('records'),list)):
                raise ValueError('Invalid pixel resume cache')
        except (OSError,ValueError,TypeError) as exc:
            cache['diagnostics'].append(f'{path.name}: {exc}; sidecar cache ignored')
            return cache
        verified={}
        for key,protocol in saved['protocols'].items():
            try:
                if not isinstance(protocol,dict) or _protocol_key(protocol)!=key:
                    raise ValueError('protocol fingerprint mismatch')
                verified[key]=protocol
            except (TypeError,ValueError) as exc:
                cache['diagnostics'].append(f'{path.name}: {exc}; protocol {key} ignored')
        for index,record in enumerate(saved['records']):
            try:
                if not isinstance(record,dict):raise ValueError('record must be an object')
                protocol_key=record.get('protocol_key')
                if not isinstance(protocol_key,str) or protocol_key not in verified:
                    raise ValueError('record has no verified protocol')
                scores=record.get('scores')
                if not isinstance(scores,dict):raise ValueError('scores must be an object')
                values={name:scores[name] for name in METRICS if name in scores}
                row={**record,**values,'metric_status':dict.fromkeys(values,'ok')}
                merge_pixel_cache(cache,verified[protocol_key],[row],source=f'{path.name} record {index}')
            except (TypeError,ValueError,KeyError) as exc:
                cache['diagnostics'].append(f'{path.name} record {index}: {exc}; ignored')
    return cache


def cached_scores(cache,protocol,row):
    scores={};binding=row['input_binding']
    # Alpha policy text is diagnostic. RGB/raw hashes, geometry and all other
    # provenance still have to match after live output validation above.
    identity=lambda value:{k:v for k,v in value.items() if k!='alpha_policy'}
    for key,previous in cache['protocols'].items():
        record=cache['records'].get((key,row['id'],binding['sha256']))
        if (record and record['input_binding']==binding and identity(record['provenance'])==identity(row['provenance'])
                and _compatible_protocol(previous,protocol)):
            scores.update(record['scores'])
    return scores


def summarize(results):
    metric_coverage={}
    means={}
    for name in METRICS:
        states=[r['metric_status'][name] for r in results]
        valid=states.count('ok');pending=states.count('pending')
        complete=bool(results) and valid==len(results)
        metric_coverage[name]={
            'expected':len(results),'valid':valid,'failed':len(results)-valid-pending,
            'pending':pending,'coverage':valid/len(results) if results else 0.0,
            'complete':complete,
        }
        # Same arithmetic mean as the official implementation. Each metric
        # needs its own full requested coverage; partial-case status must not
        # discard otherwise valid SSIM/LPIPS after an alignment failure.
        means['avg_'+name]=(float(np.mean([float(r[name]) for r in results]))
                            if complete else None)
    good=[r for r in results if r['status']=='ok']
    pending=sum(r['status']=='pending' for r in results)
    base={'expected':len(results),'valid':len(good),'failed':len(results)-len(good)-pending,'pending':pending,
          'partial':sum(r['status']=='partial' for r in results),
          'coverage':len(good)/len(results) if results else 0.0,
          'complete':bool(results) and all(c['complete'] for c in metric_coverage.values()),
          'metrics':means,'metric_coverage':metric_coverage}
    if not base['complete']:
        base['reason']='Each mean requires complete coverage for that metric; incomplete-metric means are null'
    return base


def _failed_row(row,exc,stage):
    reason=f'{type(exc).__name__}: {exc}'
    row.update({name:None for name in METRICS})
    row.update(status='error',error=reason,failure_stage=stage,
               metric_status={name:'error' for name in METRICS},
               metric_errors={name:reason for name in METRICS})


def _scored_row(row,scores):
    """Validate the helper contract before accepting any metric as successful."""
    states=scores['metric_status'];errors=scores['metric_errors']
    if set(states)!=set(METRICS) or set(errors)-set(METRICS):
        raise ValueError('Invalid official metric status/error keys')
    for name in METRICS:
        if states[name]=='ok':
            value=scores[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)):
                raise ValueError(f'Invalid successful {name} value')
            if not math.isfinite(value) and not (name=='psnr' and value==math.inf):
                raise ValueError(f'Non-finite successful {name} value')
            if name in errors:raise ValueError(f'Successful {name} has an error')
        elif states[name]=='error':
            if scores[name] is not None or not isinstance(errors.get(name),str) or not errors[name]:
                raise ValueError(f'Failed {name} needs a null score and error reason')
        else:raise ValueError(f'Invalid {name} state: {states[name]}')
    row.update({name:scores[name] for name in METRICS},metric_status=dict(states),metric_errors=dict(errors))
    valid=sum(value=='ok' for value in states.values())
    row['status']='ok' if valid==len(METRICS) else 'partial' if valid else 'error'
    if errors:
        row.update(failure_stage='metrics',error='; '.join(name+': '+errors[name] for name in METRICS if name in errors))


def _json_safe(value):
    if isinstance(value,float) and math.isinf(value):return 'inf' if value>0 else '-inf'
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [_json_safe(v) for v in value]
    return value


def _dump(path,value):
    write_json(path,_json_safe(value))


def _load_tasks(dataset_root,ids):
    tasks=[json.loads(l) for l in (dataset_root/'tasks.jsonl').read_text().splitlines() if l.strip()]
    known={}
    for task in tasks:
        if not isinstance(task.get('id'),str) or task['id'] in known:raise ValueError(f'Invalid/duplicate dataset task id: {task.get("id")}')
        known[task['id']]=task
    if ids is None:
        if not tasks:raise ValueError('Dataset has no tasks')
        return tasks,known
    requested=[s.strip() for s in ids.split(',')]
    if not requested or any(not s for s in requested):raise ValueError('--ids needs nonempty case IDs')
    if len(set(requested))!=len(requested):raise ValueError('Duplicate IDs in --ids')
    unknown=[s for s in requested if s not in known]
    if unknown:raise ValueError('Unknown requested IDs: '+', '.join(unknown))
    return [known[s] for s in requested],known


def _configure_runtime(args):
    import cv2
    cv2.setNumThreads(args.cpu_threads)
    try:
        import torch
        torch.set_num_threads(args.cpu_threads)
    except Exception as exc:
        if args.device=='cuda':
            raise ValueError('CUDA requested but PyTorch runtime could not initialize') from exc
        # MSE/PSNR do not require torch. Let the independent perceptual path
        # report its actual error instead of rejecting all four metrics here.
        return [f'PyTorch runtime configuration failed: {type(exc).__name__}: {exc}']
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested without an allocated visible GPU')
    return []


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--predictions',type=Path,help='Native RGB <id>.png directory')
    inputs.add_argument('--generation-manifest',type=Path,help='Verified raw/RGB canvas/ROI records')
    parser.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    parser.add_argument('--output',type=Path,default=Path('scores.json'))
    parser.add_argument('--ids',help='Comma-separated case IDs; unknown/duplicates fail')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu',help='Official SSIM/LPIPS compute device; CUDA requires a GPU allocation')
    parser.add_argument('--official-evaluation-dir',type=Path,default=OFFICIAL_EVALUATION)
    parser.add_argument('--cpu-threads',type=int,default=2,help='CPU thread limit for SIFT/PyTorch; does not change metric formulas')
    parser.add_argument('--no-resume',action='store_false',dest='resume',help='Recompute successful metrics as well; default reuses validated successes')
    args=parser.parse_args(argv)
    args.output=args.output.resolve()
    try:
        with output_lock(args.output):return _evaluate(args)
    except Exception as exc:
        error=args.output.with_name(args.output.name+'.error.json')
        _dump(error,{'status':'invalid_request','fatal_errors':[str(exc)],'evaluation_output_preserved':True})
        print(str(exc),file=sys.stderr);return 2


def _evaluate(args):
    protocol={'version':PROTOCOL_VERSION}
    try:
        if args.cpu_threads<1:raise ValueError('--cpu-threads must be positive')
        tasks,known=_load_tasks(args.dataset_root,args.ids)
        track=dataset_track(tasks)
        records=load_generation_records(args.generation_manifest) if args.generation_manifest else {}
        unknown=sorted(set(records)-set(known))
        if unknown:raise ValueError('Generation manifest has unknown dataset IDs: '+', '.join(unknown))
        protocol.update(protocol_info(args.official_evaluation_dir))
        if track=='asset':
            protocol.update(version=ASSET_PIXEL_PROTOCOL,track='asset',
                comparison='Prediction versus source outside the source/instruction allowed-edit region.',
                mask_policy='references.jsonl pixel_preservation_mask; unique GT target region is not a scoring mask')
        versions={}
        for package in ('numpy','opencv-python','opencv-python-headless','torch','torchvision','torchmetrics','tabulate'):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:pass
        protocol.update(device=args.device,cpu_threads=args.cpu_threads,software_versions=versions,
            artifact_binding_version=binding_version(args.dataset_root),dataset=dataset_identity(args.dataset_root),
            evaluator_sources=pixel_sources(),
            prediction_adaptation='Verified native RGB or explicit right/bottom-padding ROI before official metric calculations; raw files unchanged',
            mask_inputs=['pixel_preservation_mask'] if track=='asset' else ['mask.png','target_region_mask.png'],
            completion_note='Official pixel metrics measure preservation, not instruction success. MLLM scoring remains separate.',
            coverage_policy='Every metric has its own requested denominator, valid/failed/pending counts and coverage. Only that metric\'s incomplete mean is null; any incomplete metric makes the overall run fail.',
            case_counts='valid counts cases with all four metrics; failed includes partial cases with at least one failed metric; partial is a subset of failed.',
            json_infinity='Official positive infinity is preserved in means and represented by string inf in strict JSON.')
    except Exception as exc:
        result={'status':'invalid_request','protocol':protocol,'attempted':0,'valid':0,'failed':0,
                'complete':False,'coverage':0.0,'fatal_errors':[str(exc)],'summary':None,'results':[]}
        error=args.output.with_name(args.output.name+'.error.json') if args.output.exists() else args.output
        _dump(error,result);print(str(exc),file=sys.stderr);return 2
    cache=load_pixel_cache(args.output)
    for path in (args.output,cache_path(args.output)):
        if path.exists():shutil.copy2(path,path.with_name(path.name+'.bak'))
    resume={'enabled':args.resume,'reused_metrics':0,'computed_metrics':0,'skipped_cases':0}
    results=[{'id':t['id'],'operation':t.get('operation','unknown'),'domain':t.get('domain','unknown'),
              'status':'pending',**{name:None for name in METRICS},
              'metric_status':{name:'pending' for name in METRICS},'metric_errors':{}} for t in tasks]
    def checkpoint(persist_cache=True):
        if persist_cache:
            merge_pixel_cache(cache,protocol,results)
            _dump(cache_path(args.output),{'schema_version':1,'protocols':cache['protocols'],'records':list(cache['records'].values())})
        summary=summarize(results)
        groups=lambda field:{key:summarize([r for r in results if r[field]==key]) for key in sorted({r[field] for r in results})}
        status='running' if summary['pending'] else 'complete' if summary['complete'] else 'incomplete'
        output={'status':status,'protocol':protocol,'attempted':len(results)-summary['pending'],
            'expected':len(results),'valid':summary['valid'],'failed':summary['failed'],'partial':summary['partial'],'pending':summary['pending'],
            'coverage':summary['coverage'],'complete':summary['complete'],'summary':summary,
            'by_operation':groups('operation'),'by_domain':groups('domain'),'results':results,'resume':dict(resume),
            'resume_diagnostics':list(cache['diagnostics'])}
        _dump(args.output,output);return output
    checkpoint()
    initialized=False
    for task,row in zip(tasks,results):
        stage='input'
        kept={};needed=list(METRICS)
        try:
            prediction,provenance=load_prediction(task,args.dataset_root,predictions_dir=args.predictions,
                record=records.get(task['id']),manifest_path=args.generation_manifest)
            folder=args.dataset_root/'cases'/task['id']
            with Image.open(args.dataset_root/task['image']) as image:
                if image.mode!='RGB':raise ValueError('Source image must be RGB')
                source=np.asarray(image).copy()
            masks=[]
            mask_paths=[preservation_mask_path(task,args.dataset_root)] if track=='asset' else [folder/name for name in ('mask.png','target_region_mask.png')]
            for mask_path in mask_paths:
                with Image.open(mask_path) as image:mask=np.asarray(image).copy()
                if mask.ndim!=2 or mask.shape!=source.shape[:2] or mask.dtype!=np.uint8 or not np.all((mask==0)|(mask==255)):
                    raise ValueError(f'{mask_path.name} must be same-size binary grayscale')
                masks.append(mask)
            if track=='asset':masks.append(np.zeros(source.shape[:2],dtype=np.uint8))
            if prediction.shape!=source.shape:raise ValueError('Source and prediction dimensions differ')
            row['provenance']=provenance
            row['input_binding']=case_binding(task,args.dataset_root,provenance,masks=True)
            kept=cached_scores(cache,protocol,row) if args.resume else {}
            needed=[name for name in METRICS if name not in kept]
            resume['reused_metrics']+=len(kept)
            if not needed:
                _scored_row(row,{**kept,'metric_status':dict.fromkeys(METRICS,'ok'),'metric_errors':{}})
                resume['skipped_cases']+=1
                checkpoint(persist_cache=False);print(json.dumps({'id':task['id'],'status':'cached','reused_metrics':list(kept),'computed_metrics':[]}),flush=True)
                continue
            if not initialized:
                stage='runtime'
                protocol['runtime_warnings']=_configure_runtime(args) or []
                initialized=True
            stage='metrics'
            resume['computed_metrics']+=len(needed)
            kwargs={'device':args.device,'official_dir':args.official_evaluation_dir}
            if kept:kwargs['metrics']=needed
            scores=score_case(source,prediction,masks[0],masks[1],**kwargs)
            merged={**scores,**kept,'metric_status':{**scores['metric_status'],**dict.fromkeys(kept,'ok')},
                    'metric_errors':{key:value for key,value in scores['metric_errors'].items() if key not in kept}}
            _scored_row(row,merged)
        except Exception as exc:
            if kept:
                reason=f'{type(exc).__name__}: {exc}'
                _scored_row(row,{**dict.fromkeys(METRICS),**kept,
                    'metric_status':{name:'ok' if name in kept else 'error' for name in METRICS},
                    'metric_errors':{name:reason for name in METRICS if name not in kept}})
                row['failure_stage']=stage
            else:_failed_row(row,exc,stage)
        checkpoint();print(json.dumps({'id':task['id'],'status':row['status'],
            'metric_status':row['metric_status'],'metric_errors':row['metric_errors'],
            'reused_metrics':list(kept),'computed_metrics':needed if stage=='metrics' else []}),flush=True)
    summary=summarize(results);output=checkpoint()
    print(json.dumps({'status':output['status'],'attempted':len(results),'valid':summary['valid'],
        'failed':summary['failed'],'coverage':summary['coverage'],'output':str(args.output),'resume':resume}))
    return 0 if summary['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
