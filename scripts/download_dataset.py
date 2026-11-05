"""Download, verify and safely update the public PNG/JSON benchmark archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

from huggingface_hub import hf_hub_download
from PIL import Image

from settings import DATASET_ROOT

REPO_ID = 'yomi-chen/EditableDesignBench'
ARCHIVE_ROOT = 'EditableDesignBench'


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('Unsafe dataset manifest path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe dataset manifest path')
    return Path(*path.parts)


def _file(root, value):
    file = root / _relative(value)
    if not file.resolve().is_relative_to(root.resolve()) or not file.is_file() or file.is_symlink():
        raise ValueError('Missing or unsafe dataset file: ' + str(value))
    return file


def _digest(file):
    h = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _validate_track(root, expected, *, asset=False):
    tasks = _rows(root / 'tasks.jsonl')
    ids = [t['id'] for t in tasks]
    if len(tasks) != expected or len(set(ids)) != expected:
        raise ValueError(f'EditableDesignBench track requires {expected} unique tasks')
    checked = set()
    for item in _rows(root / 'checksums.jsonl'):
        relative = _relative(item['path']).as_posix()
        if relative in checked:
            raise ValueError('Duplicate dataset checksum path: ' + relative)
        file = _file(root, relative)
        if file.stat().st_size != item['bytes']:
            raise ValueError('Missing or incomplete dataset file: ' + relative)
        if _digest(file) != item['sha256']:
            raise ValueError('Dataset file differs from the published archive: ' + relative)
        checked.add(relative)
    if not checked:
        raise ValueError('Dataset checksum manifest is empty')
    references = _rows(root / 'references.jsonl')
    if len(references) != expected or {r['id'] for r in references} != set(ids):
        raise ValueError('Dataset task/reference identities differ')
    by_id = {r['id']: r for r in references}
    for task in tasks:
        case_id = task['id']
        if not isinstance(case_id, str) or _relative(case_id).as_posix() != case_id or '/' in case_id:
            raise ValueError('Unsafe dataset case identity')
        reference = by_id[case_id]
        base = 'cases/' + case_id + '/'
        fields = {'source': task['image'], 'target': reference['target'],
                  'eval_mask': reference['eval_mask'], 'target_region_mask': reference['target_region_mask'],
                  'annotation': reference['annotation']}
        source_mask = reference.get('source_mask', task.get('mask'))
        if not source_mask:
            raise ValueError('Dataset source-region mask is missing: ' + case_id)
        fields['source_mask'] = source_mask
        names = {'source': 'source.png', 'target': 'target.png', 'eval_mask': 'eval_mask.png',
                 'target_region_mask': 'target_region_mask.png', 'source_mask': 'mask.png',
                 'annotation': 'annotation.json'}
        if asset:
            fields['allowed_edit_mask'] = reference['allowed_edit_mask']
            names['allowed_edit_mask'] = 'allowed_edit_mask.png'
            if reference.get('pixel_preservation_mask') != base + 'allowed_edit_mask.png':
                raise ValueError('Asset pixel preservation must use its allowed-edit mask')
            if reference.get('gt_change_region') != base + 'eval_mask.png':
                raise ValueError('Asset construction region must use its paired GT union')
        for field, value in fields.items():
            if value != base + names[field] or value not in checked:
                raise ValueError('Unverified or mismatched paired case file: ' + str(value))
            _file(root, value)
        annotation = json.loads((root / fields['annotation']).read_text())
        if annotation.get('id') != case_id or annotation.get('instruction') != task['instruction']:
            raise ValueError('Dataset task/annotation instructions or identities differ: ' + case_id)
        size = task.get('image_size', annotation.get('size'))
        if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError('Invalid native image dimensions: ' + case_id)
        for field, value in fields.items():
            if field == 'annotation':
                continue
            with Image.open(root / value) as image:
                mode = 'RGB' if field in ('source', 'target') else 'L'
                if image.mode != mode or list(image.size) != size:
                    raise ValueError('Wrong image mode or dimensions: ' + value)
    actual = {p.relative_to(root).as_posix() for p in (root / 'cases').rglob('*') if p.is_file()}
    if actual != checked:
        raise ValueError('Dataset case files and checksum manifest differ')
    return tasks


def validate_dataset(root):
    """Validate current text200/asset200 and historical 400-text releases."""
    root = Path(root).resolve()
    index_path = root / 'dataset_index.json'
    if not index_path.exists():
        if (root / 'assets').exists():
            raise ValueError('Asset directory requires the collection dataset_index.json')
        return _validate_track(root, 400)
    index = json.loads(index_path.read_text())
    tracks = index.get('tracks', {})
    if set(tracks) != {'text', 'asset'} or index.get('default_track') != 'text':
        raise ValueError('Unsupported or incomplete dataset collection index')
    text_count = tracks['text'].get('cases')
    if type(text_count) is not int or text_count not in (200, 400) or tracks['text'].get('path') != '.':
        raise ValueError('The text track must contain 200 current or 400 historical tasks at the root')
    if tracks['asset'].get('path') != 'assets' or tracks['asset'].get('cases') != 200:
        raise ValueError('The asset track must contain 200 tasks under assets/')
    if index.get('total_cases') != text_count + 200:
        raise ValueError('Collection total differs from its complete tracks')
    text_tasks = _validate_track(root, text_count)
    if text_count == 200:
        if len({task.get('source_id') for task in text_tasks}) != 200:
            raise ValueError('Compact text track requires one task from each of 200 source designs')
        selection_path = tracks['text'].get('selection_manifest')
        if selection_path:
            selection = json.loads(_file(root, selection_path).read_text())
            selected = selection.get('selected_ids', [])
            if len(selected) != 200 or set(selected) != {t['id'] for t in text_tasks}:
                raise ValueError('Text subset selection and tasks differ')
    asset_root = root / _relative(tracks['asset']['path'])
    if not asset_root.is_dir() or asset_root.is_symlink():
        raise ValueError('Missing or unsafe asset-track directory')
    asset_tasks = _validate_track(asset_root, 200, asset=True)
    if {t['id'] for t in text_tasks} & {t['id'] for t in asset_tasks}:
        raise ValueError('Task identities collide across text and asset tracks')
    return text_tasks


def dataset_summary(root, text_tasks):
    index = Path(root) / 'dataset_index.json'
    if not index.is_file():
        return {'cases': len(text_tasks), 'tracks': {'text': len(text_tasks)}, 'release_format': 'legacy-400'}
    data = json.loads(index.read_text())
    return {'cases': data['total_cases'], 'tracks': {k: v['cases'] for k, v in data['tracks'].items()},
            'release_version': data.get('release_version'), 'release_format': 'text-and-asset-' + str(data['total_cases'])}


def _backup_path(destination):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return destination.with_name(destination.name + '.backup-' + stamp)


def install_archive(archive, destination, *, update=False, backup_path=None):
    destination = Path(destination).expanduser()
    if destination.is_symlink():
        raise ValueError('Dataset destination must be a directory, not a symbolic link')
    destination = destination.resolve()
    if destination.exists() and not update:
        raise ValueError('Destination already exists; use --update to install a verified replacement')
    if destination.exists() and not destination.is_dir():
        raise ValueError('Dataset destination is not a directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.editabledesignbench-', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            seen = set()
            for member in zipped.infolist():
                try:
                    relative = _relative(member.filename.rstrip('/'))
                except ValueError as exc:
                    raise ValueError('Unsafe archive member') from exc
                mode = stat.S_IFMT(member.external_attr >> 16)
                key = member.filename.casefold().rstrip('/')
                if (not relative.parts or relative.parts[0] != ARCHIVE_ROOT or
                        mode not in (0, stat.S_IFREG, stat.S_IFDIR) or key in seen):
                    raise ValueError('Unsafe archive member')
                seen.add(key)
            zipped.extractall(temporary)
        extracted = temporary / ARCHIVE_ROOT
        tasks = validate_dataset(extracted)
        backup = None
        if destination.exists():
            backup = Path(backup_path).resolve() if backup_path else _backup_path(destination)
            if backup.parent != destination.parent or backup.exists():
                raise ValueError('Backup must be a new directory beside the dataset destination')
            os.replace(destination, backup)
        try:
            os.replace(extracted, destination)
        except BaseException:
            if backup is not None:
                os.replace(backup, destination)
            raise
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATASET_ROOT)
    parser.add_argument('--revision', default='main', help='Hugging Face revision or commit')
    parser.add_argument('--update', action='store_true',
                        help='Verify the selected HF archive in staging, then back up and replace existing data')
    args = parser.parse_args(argv)
    destination = args.output_dir.expanduser()
    if destination.is_symlink():
        raise ValueError('Dataset destination must be a directory, not a symbolic link')
    destination = destination.resolve()
    if destination.exists() and not args.update:
        tasks = validate_dataset(destination)
        print(json.dumps({'status': 'already_present', **dataset_summary(destination, tasks),
                          'dataset_root': str(destination),
                          'update_hint': 'Pass --update to check the selected Hugging Face revision.'}))
        return 0
    archive = hf_hub_download(REPO_ID, filename='EditableDesignBench.zip', repo_type='dataset',
                              revision=args.revision, force_download=args.update)
    backup = _backup_path(destination) if destination.exists() else None
    tasks = install_archive(archive, destination, update=args.update, backup_path=backup)
    result = {'status': 'updated' if backup else 'downloaded', **dataset_summary(destination, tasks),
              'dataset_root': str(destination), 'revision': args.revision}
    if backup:
        result['backup_directory'] = str(backup)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
