"""Download planning and interrupted/access-denied behavior, with no network."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import download_models as d


class DownloadModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model_id='fixture-model'
        self.config={'repo_id':'fixture/model','local_dir':'/unused/fixture-model','pipeline':'FixturePipeline',
            'required_components':['transformer','vae','scheduler','tokenizer','text_encoder']}
        def config_for(model_id):
            if model_id!=self.model_id:raise ValueError('Unsupported fixture model')
            return dict(self.config)
        self.registry_patch=patch.object(d,'get_model_config',side_effect=config_for);self.registry_patch.start()
        self.models_patch=patch.object(d,'MODEL_CONFIGS',{self.model_id:self.config});self.models_patch.start()
        self.defaults_patch=patch.object(d,'DEFAULT_MODELS',(self.model_id,));self.defaults_patch.start()
        index={'_class_name':'FixturePipeline'}
        self.contents={'model_index.json':json.dumps(index).encode(),
            'flux-2-klein-9b.safetensors':b'DUPLICATE'}
        for name in self.config['required_components']:
            index[name]=['diffusers','FixtureClass']
            self.contents[name+'/config.json']=b'{}'
        self.contents['model_index.json']=json.dumps(index).encode()
        self.contents['transformer/a.safetensors']=b'A';self.contents['text_encoder/a.safetensors']=b'B'
        self.contents['vae/a.safetensors']=b'C'
        self.contents['transformer/model.safetensors.index.json']=json.dumps({'weight_map':{'weight':'a.safetensors'}}).encode()
        self.info=SimpleNamespace(sha='1'*40,gated='auto',
            siblings=[SimpleNamespace(rfilename=name,size=len(content),lfs=None) for name,content in self.contents.items()])
        self.api_calls=[]
        class API:
            def model_info(inner,repo,**kwargs):
                self.api_calls.append(kwargs);return self.info
        self.api=API();self.transfers=[]

    def tearDown(self):
        self.registry_patch.stop();self.models_patch.stop();self.defaults_patch.stop();self.temp.cleanup()

    def snapshot(self,**kwargs):
        self.transfers.append({key:value for key,value in kwargs.items() if key!='token'})
        target=Path(kwargs['local_dir'])
        for name in kwargs['allow_patterns']:
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(self.contents[name])

    def run_download(self,**kwargs):
        return d.download_one(self.model_id,self.root/'models',self.root/'status',
            api=self.api,token='secret-not-in-status',snapshot=self.snapshot,**kwargs)

    def test_component_plan_excludes_duplicate_root_weights(self):
        files=d.select_diffusers_files(self.info,self.config)
        self.assertNotIn('flux-2-klein-9b.safetensors',{x['path'] for x in files})
        self.assertIn('transformer/a.safetensors',{x['path'] for x in files})

    def test_all_scope_uses_only_explicit_defaults(self):
        self.assertEqual(d.selected_models('all'),[self.model_id])

    def test_modular_flux_plan_keeps_index_and_component_trees(self):
        siblings=list(self.info.siblings)+[
            SimpleNamespace(rfilename='modular_model_index.json',size=10,lfs=None),
            SimpleNamespace(rfilename='model.safetensors',size=100,lfs=None),
            SimpleNamespace(rfilename='ae.safetensors',size=20,lfs=None),
            SimpleNamespace(rfilename='variants/fp8r/model.safetensors',size=70,lfs=None)]
        info=SimpleNamespace(siblings=siblings)
        config=dict(self.config,backend='flux3-modular-local-pup')
        names={x['path'] for x in d.select_diffusers_files(info,config)}
        self.assertIn('modular_model_index.json',names)
        self.assertIn('transformer/a.safetensors',names)
        self.assertNotIn('model.safetensors',names);self.assertNotIn('ae.safetensors',names)
        self.assertNotIn('variants/fp8r/model.safetensors',names)

    def test_pup_plan_keeps_local_processor_system_prompts_and_weights(self):
        names=['config.json','model.safetensors','system_prompts.json','tokenizer.json','processor_config.json']
        info=SimpleNamespace(siblings=[SimpleNamespace(rfilename=n,size=10,lfs=None) for n in names])
        selected=d.select_diffusers_files(info,dict(self.config,backend='transformers-pup',required_components=[]))
        self.assertEqual({x['path'] for x in selected},set(names))

    def test_dependency_is_an_acquisition_but_not_a_comparison_model(self):
        self.config['dependencies']=[{'id':'pup','repo_id':'fixture/pup','local_dir':'/unused/pup','role':'prompt_upsampler'}]
        plans=d.acquisition_plans([self.model_id])
        self.assertEqual(list(plans),['pup',self.model_id])
        self.assertEqual(d.selected_models('all'),[self.model_id])
        self.assertEqual(plans['pup']['backend'],'transformers-pup')

    def test_metadata_only_never_starts_transfer_or_persists_token(self):
        report=self.run_download(metadata_only=True)
        self.assertEqual(report['status'],'ready');self.assertFalse(self.transfers)
        self.assertNotIn('secret-not-in-status',(self.root/'status/fixture-model.json').read_text())

    def test_download_pins_revision_and_validates_components(self):
        report=self.run_download()
        self.assertEqual(report['status'],'complete')
        self.assertEqual(self.transfers[0]['revision'],'1'*40)
        self.assertTrue(report['validation']['shard_references_complete'])
        self.assertEqual(report['progress']['complete_files'],report['progress']['expected_files'])
        self.assertFalse((Path(report['local_dir'])/'flux-2-klein-9b.safetensors').exists())

    def test_retry_uses_original_commit_even_if_main_has_moved(self):
        self.run_download(metadata_only=True)
        self.run_download()
        self.assertEqual(self.api_calls[0],{'files_metadata':True})
        self.assertEqual(self.api_calls[1]['revision'],'1'*40)

    def test_existing_local_model_is_reused_without_assigning_a_remote_commit(self):
        first=self.run_download();self.assertEqual(first['status'],'complete')
        calls=len(self.api_calls);transfers=len(self.transfers)
        self.config['prefer_existing']=True
        second=self.run_download()
        self.assertEqual(second['status'],'complete')
        self.assertEqual(second['acquisition'],'reused_existing_local')
        self.assertIsNone(second['revision']);self.assertFalse(second['local_commit_verified'])
        self.assertEqual(len(self.api_calls),calls);self.assertEqual(len(self.transfers),transfers)

    def test_gated_access_is_reported_without_attempting_transfer(self):
        class Denied(Exception):response=SimpleNamespace(status_code=403)
        def head(*args):raise Denied('provider message with secret-not-in-status')
        report=self.run_download(head=head)
        self.assertEqual(report['status'],'needs_access');self.assertFalse(self.transfers)
        self.assertEqual(report['access_url'],'https://huggingface.co/fixture/model')
        state=(self.root/'status/fixture-model.json').read_text()
        self.assertNotIn('secret-not-in-status',state)

    def test_missing_or_wrong_size_file_is_not_declared_complete(self):
        def incomplete(**kwargs):self.snapshot(**kwargs);(Path(kwargs['local_dir'])/'transformer/a.safetensors').unlink()
        report=d.download_one(self.model_id,self.root/'models',self.root/'status',
            api=self.api,token='x',snapshot=incomplete)
        self.assertEqual(report['status'],'error')

    def test_missing_index_shard_is_rejected(self):
        name='transformer/model.safetensors.index.json'
        self.contents[name]=json.dumps({'weight_map':{'weight':'missing.safetensors'}}).encode()
        for f in self.info.siblings:
            if f.rfilename==name:f.size=len(self.contents[name])
        report=self.run_download()
        self.assertEqual(report['status'],'error')

    def test_invalid_scope_and_unsafe_repo_paths_fail(self):
        for scope in ('','all,fixture-model','fixture-model,fixture-model','unknown'):
            with self.assertRaises(ValueError):d.selected_models(scope)
        self.info.siblings.append(SimpleNamespace(rfilename='../outside',size=1))
        with self.assertRaisesRegex(ValueError,'Unsafe'):d.select_diffusers_files(self.info,self.config)


if __name__=='__main__':unittest.main()
