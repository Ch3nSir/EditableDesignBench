"""Download installation boundaries; no network or model/API is used."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from download_dataset import install_archive, validate_dataset


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / 'data.zip'

    def tearDown(self):
        self.temp.cleanup()

    def test_path_traversal_is_rejected_before_extraction(self):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('EditableDesignBench/../../escape', b'bad')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            install_archive(self.archive, self.root / 'output')
        self.assertFalse((self.root / 'escape').exists())
        self.assertFalse((self.root / 'output').exists())

    def test_symlink_members_are_rejected(self):
        item = zipfile.ZipInfo('EditableDesignBench/link')
        item.create_system = 3
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr(item, '/outside')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            install_archive(self.archive, self.root / 'output')

    def test_existing_directory_is_preserved(self):
        output = self.root / 'output'
        output.mkdir()
        marker = output / 'user-file'
        marker.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            install_archive(self.archive, output)
        self.assertEqual(marker.read_text(), 'keep')

    def test_changed_reference_file_is_rejected(self):
        root = self.root / 'dataset'
        root.mkdir()
        tasks = [{'id': str(i), 'image': 'source.png'} for i in range(400)]
        (root / 'tasks.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in tasks))
        (root / 'source.png').write_bytes(b'changed')
        (root / 'checksums.jsonl').write_text(json.dumps({
            'path': 'source.png', 'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}) + '\n')
        with self.assertRaisesRegex(ValueError, 'differs from the published archive'):
            validate_dataset(root)


if __name__ == '__main__':
    unittest.main()
