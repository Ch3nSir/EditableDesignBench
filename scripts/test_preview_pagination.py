"""Paginated previews retain complete bindings without giant image canvases."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from output_adapter import prepare_rgb_record,sha256_file,instruction_sha256,load_prediction
from evaluation_binding import preview_sidecar
import preview_run
from smoke_report import _preview_inputs_match


class PreviewPaginationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.dataset=self.root/'dataset';self.run=self.root/'run'
        self.dataset.mkdir();self.run.mkdir();self.output=self.run/'comparison.png'
        self.tasks=[];self.records=[];self.manifest=self.run/'generation_manifest.json'

    def tearDown(self):self.temp.cleanup()

    def fixture(self,count):
        image=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
        for number in range(count):
            cid=f'case_{number:03}';folder=self.dataset/'cases'/cid;folder.mkdir(parents=True)
            for name in ('source.png','target.png'):Image.fromarray(image).save(folder/name)
            box={'x':5,'y':5,'width':20,'height':15}
            (folder/'annotation.json').write_text(json.dumps({'source_geometry':{'box':box},'target_geometry':{'box':box}}))
            task={'id':cid,'image':f'cases/{cid}/source.png','instruction':"Delete 'A'.",'operation':'delete'}
            self.tasks.append(task)
            raw=self.run/f'{cid}.png';Image.fromarray(image).save(raw)
            metadata={'id':cid,'source_size':[64,64],'canvas_size':[64,64],'evaluation_roi':[0,0,64,64],
                'padding':{'left':0,'top':0,'right':0,'bottom':0,'mode':'none'},
                'source_sha256':sha256_file(folder/'source.png'),'instruction_sha256':instruction_sha256(task)}
            self.records.append(prepare_rgb_record(raw,raw,metadata))
        (self.dataset/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))
        self.manifest.write_text(json.dumps({'schema_version':1,'records':self.records}))

    def render(self,page_size=20):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preview_run.main(['--dataset-root',str(self.dataset),
                '--generation-manifest',str(self.manifest),'--output',str(self.output),
                '--page-size',str(page_size)]),0)
        return json.loads(preview_sidecar(self.output).read_text())

    def validate(self):
        provenances={}
        for task,record in zip(self.tasks,self.records):
            _,provenances[task['id']]=load_prediction(task,self.dataset,record=record,manifest_path=self.manifest)
        _preview_inputs_match(self.output,[t['id'] for t in self.tasks],self.dataset,provenances)

    def test_short_preview_keeps_original_output_name_and_single_image(self):
        self.fixture(7);metadata=self.render()
        self.assertEqual(len(metadata['pages']),1)
        self.assertEqual(metadata['pages'][0]['path'],'comparison.png')
        with Image.open(self.output) as image:self.assertEqual(image.size,(1290,7*235))
        self.assertTrue(self.output.with_suffix('.html').is_file())
        self.validate()

    def test_multiple_pages_cover_every_case_and_primary_is_first_page(self):
        self.fixture(5);metadata=self.render(2)
        self.assertEqual([len(p['case_ids']) for p in metadata['pages']],[2,2,1])
        self.assertEqual(self.output.read_bytes(),(self.run/metadata['pages'][0]['path']).read_bytes())
        for page in metadata['pages']:
            with Image.open(self.run/page['path']) as image:self.assertLessEqual(image.height,470)
        self.assertEqual(len(metadata['results']),5)
        self.validate()

    def test_page_and_index_tampering_are_rejected(self):
        self.fixture(3);metadata=self.render(2)
        page=self.run/metadata['pages'][1]['path'];before=page.read_bytes()
        Image.new('RGB',(10,10),'red').save(page)
        with self.assertRaisesRegex(ValueError,'page/index changed'):self.validate()
        page.write_bytes(before)
        self.output.with_suffix('.html').write_text('old index')
        with self.assertRaisesRegex(ValueError,'page/index changed'):self.validate()

    def test_missing_coverage_or_escaping_page_path_is_rejected(self):
        self.fixture(3);metadata=self.render(2)
        metadata['pages'][1]['case_ids']=[]
        preview_sidecar(self.output).write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'coverage differs'):self.validate()
        metadata=self.render(2);metadata['pages'][0]['path']='../outside.png'
        preview_sidecar(self.output).write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'escapes'):self.validate()

    def test_last_page_reference_change_is_not_hidden_by_first_page(self):
        self.fixture(3);self.render(2)
        Image.new('RGB',(64,64),'red').save(self.dataset/'cases/case_002/target.png')
        with self.assertRaisesRegex(ValueError,'Preview inputs changed'):self.validate()

    def test_four_hundred_cases_create_twenty_bounded_pages(self):
        self.fixture(400);metadata=self.render()
        self.assertEqual(len(metadata['pages']),20)
        self.assertEqual(len(metadata['results']),400)
        self.assertEqual(sum(len(p['case_ids']) for p in metadata['pages']),400)
        for page in metadata['pages']:
            with Image.open(self.run/page['path']) as image:self.assertLessEqual(image.height,20*235)


if __name__=='__main__':unittest.main()
