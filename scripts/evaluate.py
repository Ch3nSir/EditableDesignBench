"""Local TextEditBench official MSE/PSNR/SSIM/LPIPS with verified output ROI.

Metric math and successful-case means reuse the unmodified official source.
Missing/invalid outputs still fail explicitly; model output files stay unchanged.
"""
from settings import DATASET_ROOT, MODEL_ROOT
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from output_adapter import load_generation_records, load_prediction
from evaluation_binding import BINDING_VERSION, case_binding, dataset_identity, pixel_sources
from official_pixel_metrics import (
    METRICS, PROTOCOL_VERSION, OFFICIAL_EVALUATION, protocol_info, score_case,
)

ROOT=Path(__file__).resolve().parents[1]


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
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(_json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


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
    args=parser.parse_args(argv)
    protocol={'version':PROTOCOL_VERSION}
    try:
        if args.cpu_threads<1:raise ValueError('--cpu-threads must be positive')
        tasks,known=_load_tasks(args.dataset_root,args.ids)
        records=load_generation_records(args.generation_manifest) if args.generation_manifest else {}
        unknown=sorted(set(records)-set(known))
        if unknown:raise ValueError('Generation manifest has unknown dataset IDs: '+', '.join(unknown))
        protocol.update(protocol_info(args.official_evaluation_dir))
        versions={}
        for package in ('numpy','opencv-python','opencv-python-headless','torch','torchvision','torchmetrics','tabulate'):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:pass
        protocol.update(device=args.device,cpu_threads=args.cpu_threads,software_versions=versions,
            artifact_binding_version=BINDING_VERSION,dataset=dataset_identity(args.dataset_root),
            evaluator_sources=pixel_sources(),
            prediction_adaptation='Verified native RGB or explicit right/bottom-padding ROI before official metric calculations; raw files unchanged',
            mask_inputs=['mask.png','target_region_mask.png'],
            completion_note='Official pixel metrics measure preservation, not instruction success. MLLM scoring remains separate.',
            coverage_policy='Every metric has its own requested denominator, valid/failed/pending counts and coverage. Only that metric\'s incomplete mean is null; any incomplete metric makes the overall run fail.',
            case_counts='valid counts cases with all four metrics; failed includes partial cases with at least one failed metric; partial is a subset of failed.',
            json_infinity='Official positive infinity is preserved in means and represented by string inf in strict JSON.')
    except Exception as exc:
        result={'status':'invalid_request','protocol':protocol,'attempted':0,'valid':0,'failed':0,
                'complete':False,'coverage':0.0,'fatal_errors':[str(exc)],'summary':None,'results':[]}
        _dump(args.output,result);print(str(exc),file=sys.stderr);return 2
    results=[{'id':t['id'],'operation':t.get('operation','unknown'),'domain':t.get('domain','unknown'),
              'status':'pending',**{name:None for name in METRICS},
              'metric_status':{name:'pending' for name in METRICS},'metric_errors':{}} for t in tasks]
    def checkpoint():
        summary=summarize(results)
        groups=lambda field:{key:summarize([r for r in results if r[field]==key]) for key in sorted({r[field] for r in results})}
        status='running' if summary['pending'] else 'complete' if summary['complete'] else 'incomplete'
        output={'status':status,'protocol':protocol,'attempted':len(results)-summary['pending'],
            'expected':len(results),'valid':summary['valid'],'failed':summary['failed'],'partial':summary['partial'],'pending':summary['pending'],
            'coverage':summary['coverage'],'complete':summary['complete'],'summary':summary,
            'by_operation':groups('operation'),'by_domain':groups('domain'),'results':results}
        _dump(args.output,output);return output
    checkpoint()
    initialized=False
    for task,row in zip(tasks,results):
        stage='input'
        try:
            prediction,provenance=load_prediction(task,args.dataset_root,predictions_dir=args.predictions,
                record=records.get(task['id']),manifest_path=args.generation_manifest)
            folder=args.dataset_root/'cases'/task['id']
            with Image.open(args.dataset_root/task['image']) as image:
                if image.mode!='RGB':raise ValueError('Source image must be RGB')
                source=np.asarray(image).copy()
            masks=[]
            for filename in ('mask.png','target_region_mask.png'):
                with Image.open(folder/filename) as image:mask=np.asarray(image).copy()
                if mask.ndim!=2 or mask.shape!=source.shape[:2] or mask.dtype!=np.uint8 or not np.all((mask==0)|(mask==255)):
                    raise ValueError(f'{filename} must be same-size binary grayscale')
                masks.append(mask)
            if prediction.shape!=source.shape:raise ValueError('Source and prediction dimensions differ')
            row['provenance']=provenance
            row['input_binding']=case_binding(task,args.dataset_root,provenance,masks=True)
            if not initialized:
                stage='runtime'
                protocol['runtime_warnings']=_configure_runtime(args) or []
                initialized=True
            stage='metrics'
            _scored_row(row,score_case(source,prediction,masks[0],masks[1],device=args.device,
                official_dir=args.official_evaluation_dir))
        except Exception as exc:_failed_row(row,exc,stage)
        checkpoint();print(json.dumps({'id':task['id'],'status':row['status'],
            'metric_status':row['metric_status'],'metric_errors':row['metric_errors']}),flush=True)
    summary=summarize(results);output=checkpoint()
    print(json.dumps({'status':output['status'],'attempted':len(results),'valid':summary['valid'],
        'failed':summary['failed'],'coverage':summary['coverage'],'output':str(args.output)}))
    return 0 if summary['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
