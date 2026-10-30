"""Download and safely unpack the public PNG/JSON benchmark archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from huggingface_hub import hf_hub_download

from settings import DATASET_ROOT

REPO_ID = 'yomi-chen/EditableDesignBench'


def validate_dataset(root):
    root = Path(root)
    tasks = [json.loads(line) for line in (root / 'tasks.jsonl').read_text().splitlines() if line.strip()]
    if len(tasks) != 400 or len({t['id'] for t in tasks}) != 400:
        raise ValueError('EditableDesignBench v1 requires 400 unique tasks')
    for line in (root / 'checksums.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe dataset manifest path')
        file = root / relative
        if not file.is_file() or file.stat().st_size != item['bytes']:
            raise ValueError('Missing or incomplete dataset file: ' + item['path'])
        h = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != item['sha256']:
            raise ValueError('Dataset file differs from the published archive: ' + item['path'])
    references = [json.loads(line) for line in (root / 'references.jsonl').read_text().splitlines() if line.strip()]
    if {r['id'] for r in references} != {t['id'] for t in tasks}:
        raise ValueError('Dataset task/reference identities differ')
    for task in tasks:
        if not (root / task['image']).is_file():
            raise ValueError('Task source image is missing: ' + task['id'])
    return tasks


def install_archive(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Destination already exists; choose an empty new output directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.editabledesignbench-', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                relative = Path(member.filename)
                if (relative.is_absolute() or '..' in relative.parts or
                        not relative.parts or relative.parts[0] != 'EditableDesignBench' or
                        stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError('Unsafe archive member')
            zipped.extractall(temporary)
        extracted = temporary / 'EditableDesignBench'
        tasks = validate_dataset(extracted)
        extracted.replace(destination)
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATASET_ROOT)
    parser.add_argument('--revision', default='main', help='Hugging Face revision or commit')
    args = parser.parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        tasks = validate_dataset(destination)
        print(json.dumps({'status': 'already_present', 'cases': len(tasks), 'dataset_root': str(destination)}))
        return 0
    archive = hf_hub_download(REPO_ID, filename='EditableDesignBench.zip', repo_type='dataset', revision=args.revision)
    tasks = install_archive(archive, destination)
    print(json.dumps({'status': 'downloaded', 'cases': len(tasks), 'dataset_root': str(destination)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
