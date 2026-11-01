"""Offline adapter, identity and resume contracts for registered generators."""
import builtins
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import generate_models as runner
from evaluation_binding import canonical_hash
from model_registry import DEFAULT_MODELS,MODEL_CONFIGS,get_model_config,inference_kwargs,validate_pipeline_signature


class FakeGenerator:
    def __init__(self,device):self.device=device
    def manual_seed(self,seed):self.seed=seed;return self


class IdentityPipeline:
    vae_scale_factor=8
    def __init__(self):self.calls=[]
    def __call__(self,**kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[kwargs['image'].copy()])


class MultiGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.dataset=self.root/'dataset';self.dataset.mkdir();self.output=self.root/'run'
        self.tasks=[]
        for cid,color in [('a','white'),('b','blue')]:
            folder=self.dataset/'cases'/cid;folder.mkdir(parents=True)
            Image.new('RGB',(35,49),color).save(folder/'source.png')
            # Private reference deliberately differs and must never be read by
            # generation; the fake pipeline sees only source pixels.
            Image.new('RGB',(35,49),'red').save(folder/'target.png')
            self.tasks.append({'id':cid,'image':f'cases/{cid}/source.png',
                               'instruction':f"Delete '{cid}'.",'operation':'delete'})
        (self.dataset/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))
        self.model=self.root/'checkpoint';self.model.mkdir()
        self.config=get_model_config('qwen-edit-2511');self.configure_model(self.config)
        self.torch=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda device:'NVIDIA H200 fixture',empty_cache=lambda:None),
                                   Generator=FakeGenerator,inference_mode=contextlib.nullcontext)

    def tearDown(self):self.temp.cleanup()

    def configure_model(self,config):
        index={'_class_name':config['pipeline']}
        for name in config['required_components']:
            folder=self.model/name;folder.mkdir(exist_ok=True)
            (folder/'config.json').write_text('{}')
            if name in {'transformer','vae','text_encoder'} or name.startswith('text_encoder_'):
                (folder/'weights.safetensors').write_bytes(b'fixture checkpoint bytes')
            index[name]=['fixture','FixtureComponent']
        if 'is_distilled' in config:index['is_distilled']=config['is_distilled']
        (self.model/'model_index.json').write_text(json.dumps(index))
        (self.model/'transformer/weights.safetensors').write_bytes(b'fixture original weights')

    def content(self,path):
        value={'version':'FIXTURE_CONTENT_NOT_REAL_WEIGHTS',
               'model_file_sha256':runner.sha256_file(Path(path)/'transformer/weights.safetensors')}
        return {**value,'sha256':canonical_hash(value)}

    def cli(self,*extra,model_id='qwen-edit-2511'):
        return ['--model',model_id,'--dataset-root',str(self.dataset),'--model-path',str(self.model),
                '--output-dir',str(self.output),'--all',*extra]

    def preflight(self,*extra,model_id='qwen-edit-2511'):
        with patch.object(runner,'collect_generation_content',side_effect=self.content),\
                contextlib.redirect_stdout(io.StringIO()):
            return runner.main(self.cli('--preflight',*extra,model_id=model_id))

    def prepared(self):
        self.preflight()
        args=runner.parser().parse_args(self.cli())
        args.dataset_root=args.dataset_root.resolve();args.output_dir=args.output_dir.resolve()
        args.model_path=args.model_path.resolve();args.cpu_offload=self.config['cpu_offload']
        protocol=json.loads((self.output/'protocol.json').read_text())
        plan=json.loads((self.output/'input_plan.json').read_text())
        return args,protocol,plan

    def generate(self,args,protocol,plan,records=None,pipe=None):
        pipe=pipe or IdentityPipeline()
        with contextlib.redirect_stdout(io.StringIO()):
            code=runner.run_generation(args,self.config,protocol,self.tasks,plan,records or {},self.torch,
                                       pipeline_factory=lambda *a:pipe)
        manifest=json.loads((self.output/'generation_manifest.json').read_text())
        return code,manifest,pipe

    def test_registered_kwargs_match_actual_installed_pipeline_signatures(self):
        image=Image.new('RGB',(64,64),'white')
        for model_id in MODEL_CONFIGS:
            config=get_model_config(model_id)
            if config.get('backend')=='sensenova-native':
                # Optional native-package interface has dedicated source and
                # call-contract tests, independent of the core Diffusers env.
                continue
            if model_id=='flux3-klein':
                try:accepted=validate_pipeline_signature(config)
                except ValueError as exc:
                    # The preserved legacy Qwen21 environment deliberately
                    # predates modular Flux3. The new benchmark env checks its
                    # real pinned sources; absence must be a clear error.
                    self.assertIn('FLUX3',str(exc));continue
            else:accepted=validate_pipeline_signature(config)
            kwargs=inference_kwargs(config,instruction="Delete 'a'.",image=image,
                                    width=64,height=64,generator='fixture')
            self.assertEqual(set(kwargs),set(accepted))
            argument=config.get('image_argument','image')
            if argument=='images':
                self.assertEqual(len(kwargs[argument]),1);self.assertIs(kwargs[argument][0],image)
            else:self.assertIs(kwargs[argument],image)
            if model_id.startswith('qwen-image21'):
                self.assertTrue(kwargs['use_kv_cache']);self.assertEqual(kwargs['output_resolution'],2048)
                self.assertEqual(kwargs['true_cfg_scale'],1)
            else:
                self.assertNotIn('use_kv_cache',kwargs);self.assertNotIn('output_resolution',kwargs)
            if model_id.startswith('flux2'):self.assertNotIn('true_cfg_scale',kwargs)
        self.assertEqual(get_model_config('flux2-klein-9b')['num_inference_steps'],4)
        self.assertIsNone(get_model_config('flux2-dev')['call_kwargs']['caption_upsample_temperature'])
        self.assertEqual(DEFAULT_MODELS,('flux3-klein','qwen-image21','qwen-image21-turbo'))
        self.assertEqual(get_model_config('qwen-image21-turbo')['num_inference_steps'],8)

    def test_custom_profile_generates_bound_outputs_and_resumes_without_reloading(self):
        config_file=self.root/'extra models.json'
        profile={key:value for key,value in self.config.items() if key not in ('local_dir','backend')}
        config_file.write_text(json.dumps({'models':{'custom-edit':profile}}))
        with patch.dict(MODEL_CONFIGS):
            self.assertEqual(self.preflight('--model-config',str(config_file),model_id='custom-edit'),0)
            protocol=json.loads((self.output/'protocol.json').read_text())
            plan=json.loads((self.output/'input_plan.json').read_text())
            args=runner.parser().parse_args(self.cli('--model-config',str(config_file),model_id='custom-edit'))
            args.cpu_offload=False
            pipe=IdentityPipeline()
            config=get_model_config('custom-edit')
            with contextlib.redirect_stdout(io.StringIO()):
                code=runner.run_generation(args,config,protocol,self.tasks,plan,{},self.torch,
                    pipeline_factory=lambda *a:pipe)
                records=runner.require_directory(self.output,protocol)
                code2=runner.run_generation(args,config,protocol,self.tasks,plan,records,self.torch,
                    pipeline_factory=lambda *a:self.fail('Valid custom outputs must be reused'))
            self.assertEqual((code,code2),(0,0))
            self.assertEqual(len(pipe.calls),2)
            self.assertEqual(protocol['model_id'],'custom-edit')
            self.assertEqual(protocol['model_config']['pipeline'],self.config['pipeline'])
            for task in self.tasks:
                pixels,_=runner.load_prediction(task,self.dataset,record=records[task['id']],
                    manifest_path=self.output/'generation_manifest.json')
                with Image.open(self.dataset/task['image']) as image:
                    np.testing.assert_array_equal(pixels,np.asarray(image))
                self.assertEqual(records[task['id']]['original_instruction'],task['instruction'])

    def test_preflight_pins_distinct_model_and_content_without_torch_import(self):
        real_import=builtins.__import__
        def reject_torch(name,*args,**kwargs):
            if name=='torch' or name.startswith('torch.'):raise AssertionError('GPU/model runtime imported during preflight')
            return real_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=reject_torch):self.assertEqual(self.preflight(),0)
        protocol=json.loads((self.output/'protocol.json').read_text())
        self.assertEqual(protocol['model_id'],'qwen-edit-2511')
        self.assertEqual(protocol['run_id'],'run')
        self.assertEqual(protocol['resolution_protocol'],'native-pad32-v1')
        self.assertEqual(protocol['case_ids'],['a','b'])
        self.assertEqual(protocol['steps'],40)
        self.assertEqual(protocol['protocol_sha256'],canonical_hash({k:v for k,v in protocol.items() if k!='protocol_sha256'}))
        self.assertFalse((self.output/'generation_manifest.json').exists())

    def test_changed_weights_steps_or_case_selection_refuse_same_run(self):
        self.preflight();before=(self.output/'protocol.json').read_bytes()
        for change in ['weights','steps','selection']:
            with self.subTest(change=change):
                weight=self.model/'transformer/weights.safetensors';weight.write_bytes(b'fixture original weights')
                if change=='weights':weight.write_bytes(b'fixture modified weights')
                if change=='selection':
                    args=self.cli('--preflight');index=args.index('--all');args[index:index+1]=['--ids','a']
                    call=lambda:runner.main(args)
                else:call=lambda:self.preflight(*(['--steps','8'] if change=='steps' else []))
                with patch.object(runner,'collect_generation_content',side_effect=self.content),\
                        contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'fingerprint changed'):
                    call()
                self.assertEqual((self.output/'protocol.json').read_bytes(),before)

    def test_model_path_cannot_select_different_pipeline_or_base_variant(self):
        self.configure_model(get_model_config('flux2-klein-9b'))
        with self.assertRaisesRegex(ValueError,'pipeline mismatch'):self.preflight()
        index=json.loads((self.model/'model_index.json').read_text());index['is_distilled']=False
        (self.model/'model_index.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'is_distilled=true'):
            self.preflight(model_id='flux2-klein-9b')

    def test_batch_selection_all400_and_invalid_ids(self):
        tasks=[dict(self.tasks[0],id=f'case{i:03}') for i in range(400)]
        path=self.dataset/'tasks.jsonl';path.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        self.assertEqual(len(runner.select_tasks(self.dataset,all_cases=True)),400)
        self.assertEqual([t['id'] for t in runner.select_tasks(self.dataset,'case399,case000')],['case399','case000'])
        for ids in ('case001,case001','case001,','missing'):
            with self.subTest(ids=ids),self.assertRaises(ValueError):runner.select_tasks(self.dataset,ids)

    def test_independent_source_only_calls_and_native_padding(self):
        args,protocol,plan=self.prepared();code,manifest,pipe=self.generate(args,protocol,plan)
        self.assertEqual(code,0);self.assertEqual(manifest['status'],'complete')
        self.assertEqual(len(pipe.calls),2)
        for task,call,row in zip(self.tasks,pipe.calls,manifest['records']):
            self.assertEqual(call['prompt'],task['instruction']);self.assertEqual(call['image'].size,(64,64))
            self.assertEqual((call['width'],call['height']),(64,64))
            self.assertNotIsInstance(call['image'],list)
            self.assertNotIn('mask_image',call)
            source=np.asarray(Image.open(self.dataset/task['image']))
            self.assertTrue(np.array_equal(np.asarray(call['image'])[:49,:35],source))
            self.assertEqual(row['source_size'],[35,49]);self.assertEqual(row['evaluation_roi'],[0,0,35,49])
            self.assertEqual(row['model_id'],manifest['model_id']);self.assertEqual(row['run_id'],manifest['run_id'])
            self.assertEqual(row['protocol_sha256'],manifest['protocol_sha256'])
            self.assertEqual(row['content_provenance_sha256'],manifest['content_provenance_sha256'])
            self.assertEqual(row['packed_latent_multiple'],16)
            input_files={p.name for p in Path(row['model_input']).parent.iterdir()}
            self.assertEqual(input_files,{'source.png','task.json','model_input_padded.png'})

    def test_native_pixel_model_uses_real_grid_without_claiming_a_vae(self):
        args,protocol,plan=self.prepared()
        config=get_model_config('sensenova-u1.5-8b-mot')
        class NativePipeline:
            image_grid_factor=32
            def __call__(self,**kwargs):return SimpleNamespace(images=[kwargs['image'].copy()])
        with contextlib.redirect_stdout(io.StringIO()):
            code=runner.run_generation(args,config,protocol,self.tasks,plan,{},self.torch,
                pipeline_factory=lambda *a:NativePipeline())
        self.assertEqual(code,0)
        rows=json.loads((self.output/'generation_manifest.json').read_text())['records']
        for row in rows:
            self.assertEqual(row['image_grid_factor'],32)
            self.assertIsNone(row['vae_scale_factor']);self.assertIsNone(row['packed_latent_multiple'])

    def test_plus_images_interface_and_spatial_vae_scale_preserve_single_source(self):
        args,protocol,plan=self.prepared();config=get_model_config('joyai-image-edit-plus')
        calls=[]
        class PlusPipeline:
            vae_scale_factor_spatial=8
            def __call__(self,**kwargs):
                calls.append(kwargs)
                self_image=kwargs['images']
                if len(self_image)!=1:raise AssertionError('Extra reference images supplied')
                return SimpleNamespace(images=[self_image[0].copy()])
        with contextlib.redirect_stdout(io.StringIO()):
            code=runner.run_generation(args,config,protocol,self.tasks,plan,{},self.torch,
                pipeline_factory=lambda *a:PlusPipeline())
        self.assertEqual(code,0);self.assertEqual(len(calls),len(self.tasks))
        records=json.loads((self.output/'generation_manifest.json').read_text())['records']
        for task,call,row in zip(self.tasks,calls,records):
            self.assertEqual(call['prompt'],task['instruction']);self.assertEqual(call['images'][0].size,(64,64))
            self.assertNotIn('image',call);self.assertEqual(row['vae_scale_factor'],8)
            pixels,_=runner.load_prediction(task,self.dataset,record=row,manifest_path=self.output/'generation_manifest.json')
            with Image.open(self.dataset/task['image']) as source:np.testing.assert_array_equal(pixels,np.asarray(source))

    def test_valid_resume_skips_loading_and_corrupt_case_repairs_only_it(self):
        args,protocol,plan=self.prepared();_,manifest,_=self.generate(args,protocol,plan)
        records={r['id']:r for r in manifest['records']}
        with contextlib.redirect_stdout(io.StringIO()):
            code=runner.run_generation(args,self.config,protocol,self.tasks,plan,records,self.torch,
                                      pipeline_factory=lambda *a:self.fail('Unnecessary model load'))
        self.assertEqual(code,0)
        second_raw=Path(records['b']['raw_output']);second_before=second_raw.read_bytes()
        Path(records['a']['raw_output']).unlink()
        code,repaired,pipe=self.generate(args,protocol,plan,records)
        self.assertEqual(code,0);self.assertEqual(len(pipe.calls),1)
        self.assertEqual(pipe.calls[0]['prompt'],self.tasks[0]['instruction'])
        self.assertEqual(repaired['records'][0]['attempt'],2)
        self.assertIn('repair_reason',repaired['records'][0])
        self.assertEqual(repaired['records'][1]['attempt'],1)
        self.assertEqual(second_raw.read_bytes(),second_before)

    def test_invalid_complete_can_fail_without_loading_or_overwriting_raw(self):
        args,protocol,plan=self.prepared();_,manifest,_=self.generate(args,protocol,plan)
        records={r['id']:r for r in manifest['records']}
        raw=Path(records['a']['raw_output']);raw.write_bytes(b'corrupt fixture')
        args.repair_invalid=False
        with contextlib.redirect_stdout(io.StringIO()):
            code=runner.run_generation(args,self.config,protocol,self.tasks,plan,records,self.torch,
                                      pipeline_factory=lambda *a:self.fail('Unexpected model load'))
        self.assertEqual(code,1);self.assertEqual(raw.read_bytes(),b'corrupt fixture')
        current=json.loads((self.output/'generation_manifest.json').read_text())
        self.assertEqual(current['records'][0]['status'],'invalid_output')
        self.assertEqual(current['records'][1]['status'],'complete')

    def test_generation_lock_excludes_simultaneous_writers(self):
        with runner.generation_lock(self.output):
            with self.assertRaisesRegex(ValueError,'active writer'):
                with runner.generation_lock(self.output):self.fail('Second writer entered')

    def test_legacy_protocol_or_copied_record_is_not_overwritten(self):
        self.output.mkdir();path=self.output/'protocol.json';path.write_text('{"protocol_sha256":"legacy"}')
        before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'legacy generation protocol'):self.preflight()
        self.assertEqual(path.read_bytes(),before)
        path.unlink();args,protocol,plan=self.prepared();_,manifest,_=self.generate(args,protocol,plan)
        manifest['records'][0]['model_id']='different-model'
        (self.output/'generation_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'record identity mismatch'):
            runner.require_directory(self.output,protocol)


if __name__=='__main__':unittest.main()
