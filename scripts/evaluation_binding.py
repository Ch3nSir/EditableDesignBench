"""Bind reports to the actual evaluated inputs without changing metric math."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from output_adapter import instruction_sha256, sha256_file

BINDING_VERSION='texteditbench-artifacts-v1'


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
    return {'dataset_root':str(root),'tasks_sha256':sha256_file(root/'tasks.jsonl'),
            'checksums_sha256':sha256_file(root/'checksums.jsonl') if (root/'checksums.jsonl').exists() else None,
            'revisions':{name:summary.get(name) for name in
                         ('version','ground_truth_revision','instruction_revision')}}


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
    payload={'version':BINDING_VERSION,'id':task['id'],'task_sha256':canonical_hash(task),
             'source_sha256':source_sha,'instruction_sha256':instruction_sha,'prediction':identity}
    if masks:
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
