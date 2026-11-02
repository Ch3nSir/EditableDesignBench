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
    reports=[];errors=[]
    selected=set(models.split(',')) if isinstance(models,str) else set(models or [])
    if '' in selected:raise ValueError('Empty model selector')
    for run in discover_runs(runs_root):
        protocol=json.loads((run/'protocol.json').read_text())
        model_id=protocol.get('model_id') or Path(protocol.get('model_path') or 'unknown-model').name
        if selected and model_id not in selected:continue
        report,_=evaluate_run(run,dataset_root,mode='pixel-only' if pixel_only else 'formal',expected_ids=expected_ids)
        reports.append(report)
    if not reports:raise ValueError('No selected model runs found')
    missing=selected-{report['model_id'] for report in reports}
    if missing:raise ValueError('Unknown selected models: '+', '.join(sorted(missing)))
    identities=[(report['model_id'],report['run_id']) for report in reports]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate model_id/run_id in comparison')
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
    if len(tracks)!=1:errors.append('Text and asset tracks cannot share a comparison table')
    track=reports[0]['track']
    dimensions=ASSET_DIMENSIONS if track=='asset' else STANDARD_DIMENSIONS
    semantic_names=dimensions if track=='asset' else (*dimensions,'SE')
    fields=(*PIXEL_NAMES,*semantic_names,*(('joint_qualification_rate',) if track=='asset' else ()))
    for report in reports:
        complete=report['pixel_complete'] if pixel_only else report['full_evaluation_complete']
        metrics=report['metrics'] if not errors else dict.fromkeys(fields)
        rows.append({'track':report['track'],'model_id':report['model_id'],'run_id':report['run_id'],'expected_cases':report['expected_cases'],
            'status':report['status'],'complete':bool(complete and not errors),'pixel_complete':report['pixel_complete'],
            'semantic_complete':report['semantic_complete'],'metrics':metrics,
            'pixel_coverage':report['pixel_coverage'],'semantic_coverage':report['semantic_coverage'],
            'validation_errors':report['validation_errors'],'run_dir':report['run_dir'],
            **({'joint_qualification':report.get('joint_qualification') if not errors else None,
                'semantic_summary':report.get('semantic_summary') if not errors else None,
                'by_operation':report.get('by_operation',{}) if not errors else {}} if track=='asset' else {})})
    complete=not errors and all(row['complete'] for row in rows)
    protocol=reports[0]['comparison_protocol']
    if pixel_only:protocol={k:v for k,v in protocol.items() if k!='judge'}
    result={'version':ASSET_VERSION if track=='asset' else VERSION,'track':track,
        'semantic_dimensions':list(dimensions),
        'status':'incompatible' if errors else 'complete' if complete else 'incomplete',
        'complete':complete,'pixel_only':pixel_only,'expected_cases':reports[0]['expected_cases'],
        'scope':reports[0]['scope'],'model_runs':len(rows),'comparison_protocol':protocol,
        'comparison_protocol_sha256':canonical_hash(protocol),'validation_errors':errors,
        'results':rows,'note':'No combined score or ranking is invented. Validated complete axes remain visible independently; incomplete axes stay null and retain their requested denominators.'}
    return result,2 if errors else 0 if complete else 1


def write_csv(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    asset=result.get('track')=='asset'
    dimensions=ASSET_DIMENSIONS if asset else STANDARD_DIMENSIONS
    semantic_names=dimensions if asset else (*dimensions,'SE')
    names=(*PIXEL_NAMES,*semantic_names,*(('joint_qualification_rate',) if asset else ()))
    columns=['track','model_id','run_id','status','expected_cases','complete',*names]
    for name in PIXEL_NAMES:columns.extend([name+'_valid',name+'_failed',name+'_coverage'])
    for name in semantic_names:columns.extend([name+'_scored',name+'_expected',name+'_coverage'])
    if asset:columns.extend(['joint_qualified','joint_expected','generation_valid_outputs',
                             'generation_model_failures','generation_coverage'])
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',newline='',delete=False) as stream:
            temp=Path(stream.name);writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in result.get('results',[]):
                flat={key:row.get(key) for key in ('track','model_id','run_id','status','expected_cases','complete')}
                flat.update(row['metrics'])
                for name in PIXEL_NAMES:
                    values=row['pixel_coverage'].get(name,{})
                    flat.update({name+'_'+key:values.get(key) for key in ('valid','failed','coverage')})
                for name in semantic_names:
                    values=row['semantic_coverage'].get(name,{})
                    flat.update({name+'_'+key:values.get(key) for key in ('scored','expected','coverage')})
                if asset:
                    joint=row.get('joint_qualification') or {};summary=row.get('semantic_summary') or {}
                    flat.update(joint_qualified=joint.get('qualified'),joint_expected=joint.get('expected'),
                                generation_valid_outputs=summary.get('valid_outputs'),
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
    except Exception as exc:report={'version':VERSION,'status':'invalid_request','complete':False,'validation_errors':[str(exc)],'results':[]};code=2
    write_json(args.output_dir/'comparison.json',report);write_csv(args.output_dir/'comparison.csv',report)
    print(json.dumps({'status':report['status'],'models':len(report['results']),'output_dir':str(args.output_dir)}))
    return code


if __name__=='__main__':raise SystemExit(main())
