"""Content fingerprints for local model inputs and actually resolved Python code.

This module does not import torch, diffusers or transformers. Every collection
reads the file contents afresh, including weights; filesystem timestamps are
only used to reject concurrent mutation and never substitute for SHA-256.
"""
from __future__ import annotations

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path

PROVENANCE_VERSION = 'model-and-python-content-sha256-v1'
EXCLUDED_DIRECTORIES = frozenset({'.git', '.cache', '__pycache__'})
WEIGHT_SUFFIXES = frozenset({'.safetensors', '.bin', '.pt', '.pth', '.ckpt',
                             '.gguf', '.onnx', '.msgpack', '.h5'})


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':')).encode()).hexdigest()


def _stat_identity(path):
    info=path.stat()
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _files(root, *, python_only=False):
    """Follow model/package symlinks, but fail on cycles or broken targets."""
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError(f'Provenance directory missing: {root}')
    result=[]
    stack=[(root,frozenset())]
    while stack:
        directory,ancestors=stack.pop()
        info=directory.stat();identity=(info.st_dev,info.st_ino)
        if identity in ancestors:raise ValueError(f'Symlink directory cycle: {directory}')
        lineage=ancestors|{identity}
        with os.scandir(directory) as entries:
            for entry in entries:
                path=Path(entry.path)
                if entry.is_dir(follow_symlinks=True):
                    if entry.name not in EXCLUDED_DIRECTORIES:stack.append((path,lineage))
                elif entry.is_file(follow_symlinks=True):
                    if not python_only or path.suffix=='.py':result.append(path)
                elif entry.is_symlink():
                    raise ValueError(f'Broken or unsupported provenance symlink: {path}')
                else:raise ValueError(f'Unsupported provenance file: {path}')
    return sorted(result,key=lambda path:path.relative_to(root).as_posix())


def content_tree(root, *, python_only=False):
    """Hash actual bytes with bounded memory, including multi-GB shard files."""
    root=Path(root).resolve()
    files=_files(root,python_only=python_only)
    if not files:raise ValueError(f'No provenance files found: {root}')
    records=[];unchanged=[]
    for path in files:
        before=_stat_identity(path)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if _stat_identity(path)!=before:
            raise ValueError(f'File changed while collecting provenance: {path}')
        records.append({'path':path.relative_to(root).as_posix(),
                        'bytes':before[2],'sha256':digest.hexdigest()})
        unchanged.append((path,before))
    if files!=_files(root,python_only=python_only):
        raise ValueError(f'File inventory changed while collecting provenance: {root}')
    if any(_stat_identity(path)!=before for path,before in unchanged):
        raise ValueError(f'Files changed while collecting provenance: {root}')
    return {'root':str(root),'file_count':len(records),
            'total_bytes':sum(record['bytes'] for record in records),
            'files':records,'sha256':canonical_digest(records)}


def _package_root(name):
    # A top-level find_spec resolves the same sys.path location later used by
    # import, without executing that package or loading a model/GPU runtime.
    spec=util.find_spec(name)
    locations=list(spec.submodule_search_locations or []) if spec else []
    if len(locations)!=1:
        raise ValueError(f'Cannot pin a unique installed Python source tree: {name}')
    return Path(locations[0]).resolve()


def collect_generation_content(model_path, *, package_roots=None):
    """Return a fresh model inventory and both inference package source trees.

    package_roots is solely an injectable mapping for focused offline tests.
    Production discovers the actual import paths of diffusers/transformers.
    """
    model=content_tree(model_path)
    filenames={record['path'] for record in model['files']}
    if 'model_index.json' not in filenames:raise ValueError('Local model_index.json missing')
    if not any(Path(name).suffix in WEIGHT_SUFFIXES for name in filenames):
        raise ValueError('No local model weights found for content provenance')
    # A sharded checkpoint is not complete merely because its index exists.
    for filename in filenames:
        if not filename.endswith('.index.json'):continue
        index=json.loads((Path(model_path)/filename).read_text())
        weight_map=index.get('weight_map') if isinstance(index,dict) else None
        if not isinstance(weight_map,dict):continue
        for shard in weight_map.values():
            if not isinstance(shard,str):raise ValueError(f'Invalid shard in {filename}')
            relative=Path(filename).parent/shard
            if relative.is_absolute() or '..' in relative.parts or relative.as_posix() not in filenames:
                raise ValueError(f'Missing/invalid model shard {shard!r} in {filename}')
    packages={}
    for name in ('diffusers','transformers'):
        root=Path(package_roots[name]) if package_roots is not None else _package_root(name)
        tree=content_tree(root,python_only=True)
        try:version=metadata.version(name)
        except metadata.PackageNotFoundError:version='not_installed'
        packages[name]={**tree,'version':version,'scope':'All resolved package .py source files'}
    value={'version':PROVENANCE_VERSION,'model':model,'python_packages':packages,
           'hash_policy':'Fresh streaming SHA-256 of actual file bytes; no timestamp/hash cache',
           'excluded_directories':sorted(EXCLUDED_DIRECTORIES)}
    return {**value,'sha256':canonical_digest(value)}


def require_compatible_directory(output_dir,protocol=None):
    """Refuse legacy/incompatible artifacts without modifying their files."""
    folder=Path(output_dir);protocol_path=folder/'protocol.json'
    manifest_path=folder/'generation_manifest.json'
    previous=None
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if (previous.get('generation_protocol_version')!='qwen-image21-local-generation-v2'
                or not previous.get('content_provenance',{}).get('sha256')):
            raise ValueError('Legacy generation directory has no weight/code content fingerprint; '
                             'keep it as historical results and use a new output directory')
    elif manifest_path.exists():
        raise ValueError('Generation manifest exists without a pinned protocol; use a new output directory')
    if protocol is None:return
    fingerprint=protocol['protocol_sha256']
    if previous is not None and previous.get('protocol_sha256')!=fingerprint:
        raise ValueError('Existing output directory has a different model/code/input/protocol fingerprint; '
                         'use a new output directory')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('protocol_sha256')!=fingerprint:
            raise ValueError('Generation manifest protocol mismatch; use a new output directory')
        content_sha=protocol['content_provenance']['sha256']
        if (manifest.get('generation_protocol_version')!=protocol['generation_protocol_version']
                or manifest.get('content_provenance_sha256')!=content_sha
                or manifest.get('expected_ids')!=protocol['case_ids']):
            raise ValueError('Generation manifest content/selection provenance mismatch')
        records=manifest.get('records')
        if not isinstance(records,list):raise ValueError('Generation manifest needs records list')
        seen=set()
        for record in records:
            if (not isinstance(record,dict) or record.get('id') not in protocol['case_ids']
                    or record.get('id') in seen):
                raise ValueError('Invalid/duplicate generation manifest record ID')
            seen.add(record['id'])
            if (record.get('protocol_sha256')!=fingerprint
                    or record.get('content_provenance_sha256')!=content_sha):
                raise ValueError('Generation record protocol/content fingerprint mismatch')
