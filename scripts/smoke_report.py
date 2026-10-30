"""Combine honest generation, pixel and API-pending smoke statuses."""
from settings import DATASET_ROOT
import argparse
import json
import fcntl
from pathlib import Path
import sys

from evaluation_binding import (
    BINDING_VERSION,canonical_hash,case_binding,dataset_identity,load_tasks,pixel_sources,
    preview_sidecar,source_files,write_json,
)
from output_adapter import load_generation_records,load_prediction,sha256_file


def _pixel_structure_complete(pixel,expected):
    """Check metric coverage separately from input/file verification."""
    if not pixel or not expected or len(set(expected))!=len(expected):return False
    rows=pixel.get('results',[]);summary=pixel.get('summary') or {}
    names={'mse','psnr','ssim','lpips'}
    if not (pixel.get('protocol',{}).get('version')=='texteditbench-official-pixels-v2'
            and pixel.get('complete') is True and pixel.get('valid')==len(expected)
            and len(rows)==len(expected) and {r.get('id') for r in rows}==set(expected)
            and summary.get('complete') is True and summary.get('expected')==len(expected)
            and summary.get('valid')==len(expected)):
        return False
    means=summary.get('metrics') or {};coverage=summary.get('metric_coverage') or {}
    if set(means)!={'avg_'+name for name in names} or any(value is None for value in means.values()):return False
    if set(coverage)!=names:return False
    expected_coverage={'expected':len(expected),'valid':len(expected),'failed':0,'pending':0,'coverage':1.,'complete':True}
    if any(value!=expected_coverage for value in coverage.values()):return False
    if any(pixel.get(key)!=0 or summary.get(key)!=0 for key in ('failed','pending','partial')):return False
    return all(row.get('status')=='ok' and row.get('metric_status')==dict.fromkeys(names,'ok')
               and not row.get('metric_errors') and all(row.get(name) is not None for name in names)
               for row in rows)


def pixel_inputs_match(pixel,expected,*,dataset_root=None,manifest_path=None,provenances=None,errors=None):
    """Verify evaluated inputs/code independently of per-metric coverage."""
    errors=errors if errors is not None else []
    try:
        if not pixel or not expected or len(set(expected))!=len(expected):raise ValueError('Invalid pixel report request')
        rows=pixel.get('results')
        if not isinstance(rows,list) or len(rows)!=len(expected) or {r.get('id') for r in rows}!=set(expected):
            raise ValueError('Pixel report case IDs differ')
        protocol=pixel['protocol']
        if protocol.get('version')!='texteditbench-official-pixels-v2':raise ValueError('Pixel metric protocol changed')
        if protocol.get('artifact_binding_version')!=BINDING_VERSION:raise ValueError('Pixel report has no current input binding')
        root=Path(dataset_root or protocol['dataset']['dataset_root']).resolve()
        if protocol.get('dataset')!=dataset_identity(root):raise ValueError('Pixel dataset version changed')
        if protocol.get('evaluator_sources')!=pixel_sources():raise ValueError('Pixel evaluator source changed')
        sources=protocol.get('sources',{})
        if set(sources)!={'mse_psnr.py','lipis_ssim.py'}:raise ValueError('Missing official pixel source identities')
        for source in sources.values():
            if sha256_file(source['path'])!=source['sha256']:raise ValueError('Official pixel source changed')
        tasks=load_tasks(root)
        records=load_generation_records(manifest_path) if manifest_path else None
        for row in pixel['results']:
            task=tasks[row['id']]
            if provenances is not None:
                current=provenances[row['id']]
            elif records is not None:
                _,current=load_prediction(task,root,record=records.get(row['id']),manifest_path=manifest_path)
            elif row['provenance'].get('mode')=='recorded_padding_roi':
                path=Path(row['provenance']['manifest']);current_records=load_generation_records(path)
                _,current=load_prediction(task,root,record=current_records.get(row['id']),manifest_path=path)
            else:
                directory=Path(row['provenance']['rgb_output']).parent
                _,current=load_prediction(task,root,predictions_dir=directory)
            if row.get('provenance')!=current:raise ValueError(f'{row["id"]}: Pixel prediction provenance changed')
            if row.get('input_binding')!=case_binding(task,root,current,masks=True):
                raise ValueError(f'{row["id"]}: Pixel source/instruction/mask binding changed')
        return True
    except Exception as exc:
        errors.append(str(exc));return False


def pixel_report_complete(pixel,expected,*,dataset_root=None,manifest_path=None,provenances=None,errors=None):
    """Require all four metrics plus current inputs; retain the strict gate."""
    errors=errors if errors is not None else []
    if not _pixel_structure_complete(pixel,expected):
        errors.append('Incomplete pixel report structure/coverage');return False
    return pixel_inputs_match(pixel,expected,dataset_root=dataset_root,manifest_path=manifest_path,
                              provenances=provenances,errors=errors)


def _generation_provenances(gen,protocol,expected,dataset,manifest):
    if not gen or not expected or len(set(expected))!=len(expected):raise ValueError('Invalid generation report')
    records=load_generation_records(manifest)
    if set(records)!=set(expected) or gen.get('status')!='complete' or gen.get('expected_ids')!=expected:
        raise ValueError('Incomplete or inconsistent generation selection')
    fingerprint=protocol.get('protocol_sha256')
    if not isinstance(fingerprint,str) or gen.get('protocol_sha256')!=fingerprint:
        raise ValueError('Generation protocol mismatch')
    if canonical_hash({key:value for key,value in protocol.items() if key!='protocol_sha256'})!=fingerprint:
        raise ValueError('Generation protocol body does not match its fingerprint')
    pinned=protocol.get('generation_protocol_version')=='qwen-image21-local-generation-v2'
    if pinned:
        content=protocol['content_provenance'];content_sha=content['sha256']
        if canonical_hash({key:value for key,value in content.items() if key!='sha256'})!=content_sha:
            raise ValueError('Generation content inventory fingerprint mismatch')
        if gen.get('generation_protocol_version')!=protocol['generation_protocol_version'] or gen.get('content_provenance_sha256')!=content_sha:
            raise ValueError('Generation manifest content identity mismatch')
    tasks=load_tasks(dataset);provenances={}
    for cid in expected:
        record=records[cid]
        if record.get('protocol_sha256')!=fingerprint:raise ValueError(f'{cid}: Generation record protocol mismatch')
        if pinned and record.get('content_provenance_sha256')!=content_sha:
            raise ValueError(f'{cid}: Generation record content identity mismatch')
        _,provenances[cid]=load_prediction(tasks[cid],dataset,record=record,manifest_path=manifest)
    return provenances


def _judge_inputs_match(judge,expected,dataset,manifest,provenances):
    # Rebuild only the local input/fingerprint packaging. No request function or
    # credentials are used, so validation cannot contact a judge or spend money.
    from argparse import Namespace
    from mllm_evaluate import PROTOCOL_VERSION,PROMPT_FILE,prepare_case
    protocol=judge['protocol']
    if protocol.get('version')!=PROTOCOL_VERSION or protocol.get('prompts_sha256')!=sha256_file(PROMPT_FILE):
        raise ValueError('MLLM prompt protocol changed')
    if sha256_file(protocol['prompt_source'])!=protocol['prompt_source_sha256']:
        raise ValueError('Official MLLM prompt source changed')
    if protocol.get('mode') not in ('no_reference','reference'):raise ValueError('Invalid MLLM mode')
    args=Namespace(dataset=dataset,generation_manifest=manifest,predictions=None,mode=protocol['mode'])
    tasks=load_tasks(dataset);records=load_generation_records(manifest)
    if len(judge['results'])!=len(expected) or {r['id'] for r in judge['results']}!=set(expected):
        raise ValueError('MLLM case IDs differ')
    for row in judge['results']:
        case,_=prepare_case(tasks[row['id']],args,records,protocol)
        if row.get('fingerprint')!=case['fingerprint'] or row.get('prediction_provenance')!=provenances[row['id']]:
            raise ValueError(f'{row["id"]}: MLLM inputs changed')


def _preview_inputs_match(output,expected,dataset,provenances):
    metadata=json.loads(preview_sidecar(output).read_text())
    if not metadata.get('complete') or metadata.get('version')!=BINDING_VERSION:
        raise ValueError('Preview lacks a complete input binding')
    if metadata.get('output_sha256')!=sha256_file(output):raise ValueError('Preview image changed')
    if metadata.get('dataset')!=dataset_identity(dataset):raise ValueError('Preview dataset version changed')
    if metadata.get('sources')!=source_files(('preview_run.py','output_adapter.py','evaluation_binding.py')):
        raise ValueError('Preview renderer changed')
    pages=metadata.get('pages')
    if metadata.get('preview_format')!='paged-comparisons-v1' or not isinstance(pages,list) or not pages:
        raise ValueError('Preview lacks page bindings')
    directory=Path(output).resolve().parent;page_ids=[]
    def check_artifact(item):
        if not isinstance(item,dict) or not isinstance(item.get('path'),str):raise ValueError('Invalid preview artifact binding')
        path=(directory/item['path']).resolve()
        if not path.is_relative_to(directory):raise ValueError('Preview artifact path escapes output directory')
        if sha256_file(path)!=item.get('sha256'):raise ValueError('Preview page/index changed')
    for page in pages:
        check_artifact(page)
        if not isinstance(page.get('case_ids'),list):raise ValueError('Invalid preview page case IDs')
        page_ids.extend(page['case_ids'])
    check_artifact(metadata.get('html_index'))
    if len(page_ids)!=len(expected) or len(set(page_ids))!=len(expected) or set(page_ids)!=set(expected):
        raise ValueError('Preview page case coverage differs')
    rows=metadata['results']
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):raise ValueError('Preview case IDs differ')
    tasks=load_tasks(dataset)
    for row in rows:
        if row!=case_binding(tasks[row['id']],dataset,provenances[row['id']],preview=True):
            raise ValueError(f'{row["id"]}: Preview inputs changed')


def _update_dataset_summary(summary_path,root,passed,report,gen,records,pixel):
    """Legacy smoke metadata updates merge under a lock; formal runs stay local."""
    lock=summary_path.with_name(summary_path.name+'.lock')
    with lock.open('a+b') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            summary=json.loads(summary_path.read_text())
            item={'run_dir':str(root.resolve()),'passed':passed,
                  'report':str((root/'smoke_report.json').resolve()),'mllm_status':report['mllm_status'],
                  'pixel_metric_protocol':report['pixel_metric_protocol']}
            summary['latest_model_smoke']=item
            summary.setdefault('model_smoke_runs',{})[str(root.resolve())]=item
            summary['pixel_metric_protocol']=report['pixel_metric_protocol']
            summary['model_inference_run']=bool(gen and any(r.get('raw_output') for r in records))
            metric_coverage=((pixel or {}).get('summary') or {}).get('metric_coverage') or {}
            summary['model_scores_computed']=any(value.get('valid',0)>0 for value in metric_coverage.values())
            write_json(summary_path,summary)
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--generation-exit',required=True,type=int)
    p.add_argument('--pixel-exit',required=True,type=int)
    p.add_argument('--mllm-exit',required=True,type=int)
    p.add_argument('--preview-exit',required=True,type=int)
    p.add_argument('--no-update-dataset',action='store_true',help='Inspect reports without updating dataset summary')
    p.add_argument('--dataset-root',type=Path,help='Dataset used for current-file/hash verification')
    a=p.parse_args(argv);root=a.run_dir
    dataset=(a.dataset_root or DATASET_ROOT).resolve()
    errors=[]
    def read(name):
        f=root/name
        try:return json.loads(f.read_text()) if f.exists() else None
        except (OSError,ValueError) as exc:
            errors.append(name+': '+str(exc));return None
    protocol=read('protocol.json') or {};expected=protocol.get('case_ids',[])
    gen=read('generation_manifest.json');pixel=read('pixel_scores.json');judge=read('mllm_pending.json')
    records=gen.get('records',[]) if gen else [];provenances={}
    try:
        provenances=_generation_provenances(gen,protocol,expected,dataset,root/'generation_manifest.json')
        gen_ok=a.generation_exit==0
    except Exception as exc:
        errors.append('generation: '+str(exc));gen_ok=False
    def exact_ids(rows):
        return bool(expected) and len(rows)==len(expected) and len(set(expected))==len(expected) and {r.get('id') for r in rows}==set(expected)
    pixel_ok=bool(provenances) and a.pixel_exit==0 and pixel_report_complete(pixel,expected,
        dataset_root=dataset,manifest_path=root/'generation_manifest.json',provenances=provenances,errors=errors)
    judge_ok=bool(judge) and a.mllm_exit==0 and judge.get('status')=='not_scored' and judge.get('api_calls_made')==0 and judge.get('summary',{}).get('errors')==0 and judge.get('summary',{}).get('expected')==len(expected) and exact_ids(judge.get('results',[]))
    if judge_ok:
        try:
            if not provenances:raise ValueError('Generation input validation failed')
            _judge_inputs_match(judge,expected,dataset,root/'generation_manifest.json',provenances)
        except Exception as exc:
            errors.append('mllm: '+str(exc));judge_ok=False
    preview_ok=a.preview_exit==0 and (root/'comparison.png').is_file()
    if preview_ok:
        try:
            if not provenances:raise ValueError('Generation input validation failed')
            _preview_inputs_match(root/'comparison.png',expected,dataset,provenances)
        except Exception as exc:
            errors.append('preview: '+str(exc));preview_ok=False
    passed=gen_ok and pixel_ok and judge_ok and preview_ok
    report={'generation_and_pixel_smoke_passed':passed,'requested_cases':expected,
        'artifact_binding_version':BINDING_VERSION,'validation_errors':errors,
        'generation_identity_version':protocol.get('generation_protocol_version','legacy-unpinned'),
        'pixel_metric_protocol':pixel.get('protocol',{}).get('version') if pixel else None,
        'stages':{'generation':{'passed':gen_ok,'exit_code':a.generation_exit},
                  'pixel_scoring':{'passed':pixel_ok,'exit_code':a.pixel_exit},
                  'mllm_input_validation':{'passed':judge_ok,'exit_code':a.mllm_exit},
                  'comparison_preview':{'passed':preview_ok,'exit_code':a.preview_exit}},
        'mllm_status':judge.get('status') if judge else 'missing_report','mllm_api_calls':judge.get('api_calls_made') if judge else None,
        'mllm_real_service_verified':False,'full_semantic_evaluation_complete':False,
        'note':'Real model generation/pixel scoring and MLLM input packaging are checked. Actual MLLM judging awaits user API configuration.'}
    (root/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
    summary_path=dataset/'summary.json'
    if summary_path.exists() and not a.no_update_dataset:
        _update_dataset_summary(summary_path,root,passed,report,gen,records,pixel)
    print(json.dumps(report,indent=2));return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
