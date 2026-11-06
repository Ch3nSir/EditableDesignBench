"""Revalidate independent model runs and export comparable metric tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

from evaluation_binding import canonical_hash
from settings import DATASET_ROOT
from formal_report import ROOT, PIXEL_NAMES, ASSET_DIMENSIONS, evaluate_run, write_json
from mllm_evaluate import STANDARD_DIMENSIONS

VERSION='texteditbench-multi-model-comparison-v1'
ASSET_VERSION='editabledesignbench-assets-multi-model-comparison-v1'


def discover_runs(root):
    root=Path(root).resolve()
    if (root/'protocol.json').is_file():return [root]
    files=set(root.glob('*/protocol.json'))|set(root.glob('*/*/protocol.json'))
    return sorted({path.parent for path in files if not any(part.startswith('.') or part in ('backup','audit') for part in path.relative_to(root).parts[:-1])})


def compare_runs(runs_root,dataset_root=ROOT,*,models=None,pixel_only=False,expected_ids=None):
    reports=[];failed=[];errors=[]
    selected=set(models.split(',')) if isinstance(models,str) else set(models or [])
    if '' in selected:raise ValueError('Empty model selector')
    for run in discover_runs(runs_root):
        protocol={};model_id=run.name
        try:
            protocol=json.loads((run/'protocol.json').read_text())
            if not isinstance(protocol,dict):raise ValueError('Generation protocol must be an object')
            candidate=protocol.get('model_id') or Path(protocol.get('model_path') or run.name).name
            if not isinstance(candidate,str) or not candidate:raise ValueError('Invalid model ID')
            model_id=candidate
            if selected and model_id not in selected:continue
            report,_=evaluate_run(run,dataset_root,mode='pixel-only' if pixel_only else 'formal',expected_ids=expected_ids)
            reports.append(report)
        except Exception as exc:
            if selected and model_id not in selected:continue
            failed.append({'track':None,'model_id':model_id,'run_id':protocol.get('run_id') if isinstance(protocol,dict) else None,
                'run_dir':str(run),'status':'invalid_run','complete':False,'pixel_complete':False,
                'semantic_complete':False,'expected_cases':None,'pixel_coverage':{},'semantic_coverage':{},
                'validation_errors':[f'{type(exc).__name__}: {exc}']})
    if not reports and not failed:raise ValueError('No selected model runs found')
    missing=selected-{report['model_id'] for report in (*reports,*failed)}
    for model_id in sorted(missing):
        failed.append({'track':None,'model_id':model_id,'run_id':None,'run_dir':None,
            'status':'missing_run','complete':False,'pixel_complete':False,'semantic_complete':False,
            'expected_cases':None,'pixel_coverage':{},'semantic_coverage':{},
            'validation_errors':['No discovered run protocol for selected model: '+model_id]})
    identities=[(report['model_id'],report['run_id']) for report in reports]
    if len(identities)!=len(set(identities)):errors.append('Duplicate model_id/run_id in comparison')
    common_keys=('track','dataset','case_ids','resolution_protocol','evaluation_data_sha256','pixel')
    for key in common_keys:
        values=[report['comparison_protocol'][key] for report in reports]
        # Missing reports remain incomplete rather than inheriting another
        # model's scores. Known protocols still must agree exactly.
        known=[value for value in values if value is not None]
        if known and any(value!=known[0] for value in known[1:]):errors.append('Models differ in '+key)
    if not pixel_only:
        judges=[r['comparison_protocol']['judge'] for r in reports if r['comparison_protocol']['judge'] is not None]
        if judges and any(value!=judges[0] for value in judges[1:]):errors.append('Models differ in judge protocol')
    rows=[]
    tracks={report['track'] for report in reports}
    if len(tracks)>1:errors.append('Text and asset tracks cannot share a comparison table')
    track=next(iter(tracks)) if len(tracks)==1 else 'mixed' if tracks else None
    dimensions=(tuple(dict.fromkeys((*STANDARD_DIMENSIONS,*ASSET_DIMENSIONS))) if track=='mixed'
                else ASSET_DIMENSIONS if track=='asset' else STANDARD_DIMENSIONS)
    semantic_names=dimensions if track=='asset' else (*dimensions,'SE')
    fields=(*PIXEL_NAMES,*semantic_names,'joint_qualification_rate')
    for report in reports:
        complete=report['pixel_complete'] if pixel_only else report['full_evaluation_complete']
        metrics=report['metrics']
        rows.append({'track':report['track'],'model_id':report['model_id'],'run_id':report['run_id'],'expected_cases':report['expected_cases'],
            'status':report['status'],'complete':bool(complete),'comparable':not errors,'pixel_complete':report['pixel_complete'],
            'semantic_complete':report['semantic_complete'],'metrics':metrics,
            'comparison_protocol':report['comparison_protocol'],
            'pixel_coverage':report['pixel_coverage'],'semantic_coverage':report['semantic_coverage'],
            'validation_errors':report['validation_errors'],'run_dir':report['run_dir'],
            'joint_qualification':report.get('joint_qualification'),
            **({'semantic_summary':report.get('semantic_summary'),
                'by_operation':report.get('by_operation',{})} if report['track']=='asset' else {})})
    rows.extend({**row,'metrics':dict.fromkeys(fields),'comparable':False} for row in failed)
    complete=not errors and not failed and all(row['complete'] for row in rows)
    protocol=reports[0]['comparison_protocol'] if reports and not errors else None
    if pixel_only and protocol is not None:protocol={k:v for k,v in protocol.items() if k!='judge'}
    run_errors=[(row['run_dir'] or row['model_id'])+': '+error for row in failed for error in row['validation_errors']]
    result={'version':ASSET_VERSION if track=='asset' else VERSION,'track':track,
        'semantic_dimensions':list(dimensions),
        'status':'incompatible' if errors else 'complete' if complete else 'incomplete',
        'complete':complete,'comparable':not errors and not failed,'pixel_only':pixel_only,
        'expected_cases':reports[0]['expected_cases'] if protocol is not None else None,
        'scope':reports[0]['scope'] if protocol is not None else None,'model_runs':len(rows),'comparison_protocol':protocol,
        'comparison_protocol_sha256':canonical_hash(protocol) if protocol is not None else None,'validation_errors':errors+run_errors,
        'results':rows,'note':'No combined score or ranking is invented. Metrics and row completeness describe independent validation; incompatible protocols are not directly comparable. Incomplete axes stay null and retain their requested denominators.'}
    return result,2 if errors or failed else 0 if complete else 1


def write_csv(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    asset=result.get('track') in ('asset','mixed')
    dimensions=result.get('semantic_dimensions') or (ASSET_DIMENSIONS if asset else STANDARD_DIMENSIONS)
    semantic_names=dimensions if result.get('track')=='asset' else (*dimensions,'SE')
    names=(*PIXEL_NAMES,*semantic_names,'joint_qualification_rate')
    columns=['track','model_id','run_id','status','expected_cases','complete','comparable','comparison_errors','validation_errors',*names]
    for name in PIXEL_NAMES:columns.extend([name+'_valid',name+'_failed',name+'_coverage'])
    for name in semantic_names:columns.extend([name+'_scored',name+'_expected',name+'_coverage'])
    columns.extend(['joint_qualified','joint_expected'])
    if asset:columns.extend(['generation_valid_outputs',
                             'generation_model_failures','generation_coverage'])
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',newline='',delete=False) as stream:
            temp=Path(stream.name);writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in result.get('results',[]):
                flat={key:row.get(key) for key in ('track','model_id','run_id','status','expected_cases','complete','comparable')}
                flat['comparison_errors']='; '.join(result.get('validation_errors',[]))
                flat['validation_errors']='; '.join(row.get('validation_errors',[]))
                flat.update(row['metrics'])
                for name in PIXEL_NAMES:
                    values=row['pixel_coverage'].get(name,{})
                    flat.update({name+'_'+key:values.get(key) for key in ('valid','failed','coverage')})
                for name in semantic_names:
                    values=row['semantic_coverage'].get(name,{})
                    flat.update({name+'_'+key:values.get(key) for key in ('scored','expected','coverage')})
                joint=row.get('joint_qualification') or {}
                flat.update(joint_qualified=joint.get('qualified'),joint_expected=joint.get('expected'))
                if asset:
                    summary=row.get('semantic_summary') or {}
                    flat.update(generation_valid_outputs=summary.get('valid_outputs'),
                                generation_model_failures=summary.get('model_failures'),
                                generation_coverage=summary.get('valid_output_coverage'))
                writer.writerow(flat)
        temp.replace(path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',required=True,type=Path);parser.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    parser.add_argument('--models',help='Comma-separated model IDs; default all discovered runs')
    parser.add_argument('--pixel-only',action='store_true');parser.add_argument('--expected-ids')
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    try:report,code=compare_runs(args.runs_root,args.dataset_root,models=args.models,pixel_only=args.pixel_only,expected_ids=args.expected_ids)
    except Exception as exc:
        report={'version':VERSION,'status':'invalid_request','complete':False,'validation_errors':[str(exc)],'results':[]}
        write_json(args.output_dir/'comparison.error.json',report)
        print(json.dumps({'status':report['status'],'models':0,'output_dir':str(args.output_dir)}))
        return 2
    write_json(args.output_dir/'comparison.json',report);write_csv(args.output_dir/'comparison.csv',report)
    print(json.dumps({'status':report['status'],'models':len(report['results']),'output_dir':str(args.output_dir)}))
    return code


if __name__=='__main__':raise SystemExit(main())
