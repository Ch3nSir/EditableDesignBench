"""Bind reports to the actual evaluated inputs without changing metric math."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from output_adapter import instruction_sha256, sha256_file

BINDING_VERSION='texteditbench-artifacts-v1'
ASSET_BINDING_VERSION='editabledesignbench-assets-artifacts-v1'
ASSET_INPUT_PROTOCOL='asset-edit-single-source-v1'
ASSET_PIXEL_PROTOCOL='editabledesignbench-assets-background-pixels-v1'


def task_track(task):
    if not isinstance(task,dict):raise ValueError('Task entries must be objects')
    protocol=task.get('input_protocol')
    if protocol==ASSET_INPUT_PROTOCOL:return 'asset'
    if protocol is not None and (not isinstance(protocol,str) or not protocol.startswith('text')):
        raise ValueError('Unsupported task input protocol: '+str(protocol))
    if protocol is None and (task.get('asset_family_id') or str(task.get('id','')).startswith('asset_')):
        raise ValueError('Asset tasks must explicitly declare '+ASSET_INPUT_PROTOCOL)
    return 'text'


def dataset_track(tasks):
    values=list(tasks.values()) if isinstance(tasks,dict) else list(tasks)
    tracks={task_track(task) for task in values}
    if len(tracks)!=1:raise ValueError('Evaluate text and asset tracks separately; empty or mixed-track dataset')
    return next(iter(tracks))


def binding_version(root):
    return ASSET_BINDING_VERSION if dataset_track(load_tasks(root))=='asset' else BINDING_VERSION


def resolve_track_root(root,track=None):
    root=Path(root).expanduser().resolve()
    if track is None:return root
    if track not in ('text','asset'):raise ValueError('Unknown dataset track')
    if (root/'tasks.jsonl').is_file() and dataset_track(load_tasks(root))==track:return root
    if track=='asset' and (root/'assets/tasks.jsonl').is_file():return root/'assets'
    raise ValueError('Dataset does not contain the requested '+track+' track; download/update the dataset or set --dataset-root')


def preservation_mask_path(task,root):
    """Resolve the asset source/instruction region, never the unique GT union."""
    root=Path(root).resolve()
    if task_track(task)!='asset':raise ValueError('Allowed-region lookup is for the asset track')
    rows=[json.loads(line) for line in (root/'references.jsonl').read_text().splitlines() if line.strip()]
    refs={r['id']:r for r in rows}
    if len(refs)!=len(rows):raise ValueError('Duplicate reference case IDs')
    item=refs.get(task['id'])
    if item is None:raise ValueError('Missing asset reference: '+task['id'])
    value=item.get('pixel_preservation_mask')
    if not isinstance(value,str) or not value:raise ValueError('Asset reference requires pixel_preservation_mask')
    path=(root/value).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Invalid asset preservation mask path')
    return path


def canonical_hash(value):
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()


def source_files(names):
    folder=Path(__file__).resolve().parent
    return {name:{'path':str(folder/name),'sha256':sha256_file(folder/name)} for name in names}


def pixel_sources():
    return source_files(('evaluate.py','official_pixel_metrics.py','output_adapter.py','evaluation_binding.py'))


def dataset_identity(root):
    root=Path(root).resolve()
    summary_path=root/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    # Runtime fields in summary.json are deliberately excluded: reporting a
    # completed run must not invalidate the dataset version it just evaluated.
    identity={'dataset_root':str(root),'tasks_sha256':sha256_file(root/'tasks.jsonl'),
            'checksums_sha256':sha256_file(root/'checksums.jsonl') if (root/'checksums.jsonl').exists() else None,
            'revisions':{name:summary.get(name) for name in
                         ('version','ground_truth_revision','instruction_revision')}}
    if dataset_track(load_tasks(root))=='asset':
        identity.update(input_protocol=ASSET_INPUT_PROTOCOL,references_sha256=sha256_file(root/'references.jsonl'),
                        evaluation_protocol_sha256=sha256_file(root/'evaluation_protocol.json') if (root/'evaluation_protocol.json').exists() else None)
    return identity


def load_tasks(root):
    rows=[json.loads(line) for line in (Path(root)/'tasks.jsonl').read_text().splitlines() if line.strip()]
    tasks={row['id']:row for row in rows}
    if not tasks or len(tasks)!=len(rows):raise ValueError('Empty or duplicate dataset task IDs')
    return tasks


def case_binding(task,root,provenance,*,masks=False,preview=False):
    root=Path(root)
    source_sha=sha256_file(root/task['image']);instruction_sha=instruction_sha256(task)
    if provenance.get('source_sha256')!=source_sha or provenance.get('instruction_sha256')!=instruction_sha:
        raise ValueError('Prediction provenance source/instruction mismatch')
    identity={name:provenance.get(name) for name in
              ('mode','raw_sha256','rgb_sha256','source_size','canvas_size','evaluation_roi',
               'padding','adapter_version','validation_adapter_version','generation_protocol_sha256',
               'model_input_sha256')}
    if any(not isinstance(identity[name],str) or len(identity[name])!=64 for name in ('raw_sha256','rgb_sha256')):
        raise ValueError('Prediction provenance requires raw/RGB SHA-256')
    asset=task_track(task)=='asset'
    payload={'version':ASSET_BINDING_VERSION if asset else BINDING_VERSION,'id':task['id'],'task_sha256':canonical_hash(task),
             'source_sha256':source_sha,'instruction_sha256':instruction_sha,'prediction':identity}
    if masks:
        if asset:
            mask=preservation_mask_path(task,root)
            payload['masks']={'pixel_preservation_mask':sha256_file(mask)}
            payload['mask_files']={'pixel_preservation_mask':mask.relative_to(root.resolve()).as_posix()}
            payload['references_sha256']=sha256_file(root/'references.jsonl')
        else:
            payload['masks']={name:sha256_file(root/'cases'/task['id']/name)
                              for name in ('mask.png','target_region_mask.png')}
    if preview:
        payload['preview_files']={name:sha256_file(root/'cases'/task['id']/name)
                                  for name in ('target.png','annotation.json')}
    return {**payload,'sha256':canonical_hash(payload)}


def preview_sidecar(path):
    path=Path(path)
    return path.with_name(path.name+'.inputs.json')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name)
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
