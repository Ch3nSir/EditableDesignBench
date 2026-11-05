"""Offline archive, complete-track, update and rollback regression tests."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image

from download_dataset import dataset_summary, install_archive, main, validate_dataset


def png(mode, color):
    data = io.BytesIO()
    Image.new(mode, (2, 2), color).save(data, format='PNG')
    return data.getvalue()


def track_files(count, *, asset=False):
    files, tasks, refs, checks = {}, [], [], []
    for i in range(count):
        case_id = ('a' if asset else 't') + str(i).zfill(3)
        base = 'cases/' + case_id + '/'
        task = {'id': case_id, 'source_id': str(i), 'instruction': 'Move the item.',
                'image': base + 'source.png', 'image_size': [2, 2]}
        if not asset:
            task['mask'] = base + 'mask.png'
        ref = {'id': case_id, 'target': base + 'target.png', 'eval_mask': base + 'eval_mask.png',
               'target_region_mask': base + 'target_region_mask.png', 'annotation': base + 'annotation.json'}
        blobs = {'source.png': png('RGB', 'red'), 'target.png': png('RGB', 'blue'),
                 'mask.png': png('L', 255), 'target_region_mask.png': png('L', 255),
                 'eval_mask.png': png('L', 255),
                 'annotation.json': json.dumps({'id': case_id, 'instruction': task['instruction'], 'size': [2, 2]}).encode()}
        if asset:
            blobs['allowed_edit_mask.png'] = png('L', 255)
            ref.update(source_mask=base + 'mask.png', allowed_edit_mask=base + 'allowed_edit_mask.png',
                       pixel_preservation_mask=base + 'allowed_edit_mask.png', gt_change_region=base + 'eval_mask.png')
        else:
            blobs['task.json'] = json.dumps(task).encode()
        for name, data in blobs.items():
            relative = base + name
            files[relative] = data
            checks.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        tasks.append(task)
        refs.append(ref)
    for name, rows in [('tasks.jsonl', tasks), ('references.jsonl', refs), ('checksums.jsonl', checks)]:
        files[name] = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    return files


class DatasetDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.TemporaryDirectory()
        root = Path(cls.fixtures.name)
        cls.legacy_archive = root / 'legacy.zip'
        cls.combined_archive = root / 'combined.zip'
        cls.bad_combined_archive = root / 'bad-combined.zip'
        cls.compact_archive = root / 'compact.zip'
        text, assets = track_files(400), track_files(200, asset=True)
        combined = dict(text)
        combined.update({'assets/' + key: value for key, value in assets.items()})
        combined['dataset_index.json'] = json.dumps({
            'release_version': '2.0.0', 'total_cases': 600, 'default_track': 'text',
            'tracks': {'text': {'path': '.', 'cases': 400}, 'asset': {'path': 'assets', 'cases': 200}}
        }).encode()
        for path, files in [(cls.legacy_archive, text), (cls.combined_archive, combined)]:
            with zipfile.ZipFile(path, 'w') as z:
                for name, data in files.items():
                    z.writestr('EditableDesignBench/' + name, data)
        compact = track_files(200)
        compact.update({'assets/' + key: value for key, value in assets.items()})
        compact['dataset_index.json'] = json.dumps({
            'release_version': '3.0.0', 'total_cases': 400, 'default_track': 'text',
            'tracks': {'text': {'path': '.', 'cases': 200, 'selection_manifest': 'text_subset.json'},
                       'asset': {'path': 'assets', 'cases': 200}}
        }).encode()
        compact['text_subset.json'] = json.dumps({'selected_ids': ['t'+str(i).zfill(3) for i in range(200)]}).encode()
        with zipfile.ZipFile(cls.compact_archive, 'w') as z:
            for name, data in compact.items():z.writestr('EditableDesignBench/' + name, data)
        corrupt = dict(combined)
        corrupt['assets/cases/a000/target.png'] = png('RGB', 'green')
        with zipfile.ZipFile(cls.bad_combined_archive, 'w') as z:
            for name, data in corrupt.items():
                z.writestr('EditableDesignBench/' + name, data)

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.cleanup()

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

    def test_windows_traversal_is_rejected(self):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr(r'EditableDesignBench/..\escape', b'bad')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            install_archive(self.archive, self.root / 'output')

    def test_symlink_members_are_rejected(self):
        item = zipfile.ZipInfo('EditableDesignBench/link')
        item.create_system = 3
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr(item, '/outside')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            install_archive(self.archive, self.root / 'output')

    def test_existing_directory_is_preserved_without_update(self):
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

    def test_legacy_400_updates_to_complete_600_and_keeps_backup(self):
        destination = self.root / 'dataset'
        tasks = install_archive(self.legacy_archive, destination)
        self.assertEqual(len(tasks), 400)
        self.assertEqual(dataset_summary(destination, tasks)['cases'], 400)
        (destination / 'user-note.txt').write_text('preserve in backup')
        printed = io.StringIO()
        with mock.patch('download_dataset.hf_hub_download', return_value=str(self.combined_archive)) as download:
            with contextlib.redirect_stdout(printed):
                self.assertEqual(main(['--output-dir', str(destination), '--update']), 0)
        download.assert_called_once()
        self.assertTrue(download.call_args.kwargs['force_download'])
        report = json.loads(printed.getvalue())
        self.assertEqual(report['status'], 'updated')
        self.assertEqual(report['cases'], 600)
        self.assertEqual(report['tracks'], {'text': 400, 'asset': 200})
        backup = Path(report['backup_directory'])
        self.assertEqual((backup / 'user-note.txt').read_text(), 'preserve in backup')
        self.assertFalse((backup / 'assets').exists())
        self.assertEqual(len(validate_dataset(destination)), 400)
        printed = io.StringIO()
        with mock.patch('download_dataset.hf_hub_download') as download:
            with contextlib.redirect_stdout(printed):
                self.assertEqual(main(['--output-dir', str(destination)]), 0)
        download.assert_not_called()
        self.assertEqual(json.loads(printed.getvalue())['cases'], 600)

    def test_full_v2_updates_to_compact_v3_without_leaving_removed_cases(self):
        destination=self.root/'dataset'
        install_archive(self.combined_archive,destination)
        (destination/'old-experiment-note.txt').write_text('keep')
        backup=self.root/'v2-backup'
        tasks=install_archive(self.compact_archive,destination,update=True,backup_path=backup)
        self.assertEqual(len(tasks),200)
        self.assertEqual(dataset_summary(destination,tasks)['tracks'],{'text':200,'asset':200})
        self.assertEqual(dataset_summary(destination,tasks)['cases'],400)
        self.assertFalse((destination/'cases/t399').exists())
        self.assertEqual(len(validate_dataset(backup)),400)
        self.assertEqual((backup/'old-experiment-note.txt').read_text(),'keep')
        self.assertEqual((backup/'assets/cases/a199/target.png').read_bytes(),
                         (destination/'assets/cases/a199/target.png').read_bytes())

    def test_compact_selection_manifest_cannot_disagree_with_payload(self):
        destination=self.root/'dataset';install_archive(self.compact_archive,destination)
        (destination/'text_subset.json').write_text(json.dumps({'selected_ids':['wrong']*200}))
        with self.assertRaisesRegex(ValueError,'selection and tasks differ'):
            validate_dataset(destination)

    def test_compact_release_cannot_silently_drop_source_design_coverage(self):
        destination=self.root/'dataset';install_archive(self.compact_archive,destination)
        p=destination/'tasks.jsonl';tasks=[json.loads(l) for l in p.read_text().splitlines()]
        tasks[1]['source_id']=tasks[0]['source_id']
        p.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        with self.assertRaisesRegex(ValueError,'one task from each'):
            validate_dataset(destination)

    def test_compact_collection_total_is_verified(self):
        destination=self.root/'dataset';install_archive(self.compact_archive,destination)
        p=destination/'dataset_index.json';index=json.loads(p.read_text());index['total_cases']=600
        p.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'Collection total'):
            validate_dataset(destination)

    def test_bad_asset_checksum_never_replaces_existing_data(self):
        destination = self.root / 'dataset'
        destination.mkdir()
        (destination / 'user-file').write_text('keep')
        backup = self.root / 'backup'
        with self.assertRaisesRegex(ValueError, 'differs|incomplete'):
            install_archive(self.bad_combined_archive, destination, update=True, backup_path=backup)
        self.assertEqual((destination / 'user-file').read_text(), 'keep')
        self.assertFalse(backup.exists())

    def test_failed_final_rename_restores_original_directory(self):
        destination = self.root / 'dataset'
        destination.mkdir()
        (destination / 'user-file').write_text('keep')
        backup = self.root / 'backup'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('EditableDesignBench/validated-marker', b'new')
        original_replace = os.replace

        def fail_publish(source, target):
            if Path(source).name == 'EditableDesignBench':
                raise OSError('simulated filesystem failure')
            return original_replace(source, target)

        with mock.patch('download_dataset.validate_dataset', return_value=[]):
            with mock.patch('download_dataset.os.replace', side_effect=fail_publish):
                with self.assertRaisesRegex(OSError, 'simulated'):
                    install_archive(self.archive, destination, update=True, backup_path=backup)
        self.assertEqual((destination / 'user-file').read_text(), 'keep')
        self.assertFalse(backup.exists())


if __name__ == '__main__':
    unittest.main()
