"""Offline tests for the single-PUP modular adapter and new Qwen21 variants."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

from PIL import Image

import flux3_adapter as adapter
from model_registry import get_model_config,validate_model_files,inference_kwargs,extract_generation_output


class Flux3AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=self.root/'image';self.model.mkdir();self.pup=self.root/'pup';self.pup.mkdir()
        self.config=get_model_config('flux3-klein');self.config['dependencies'][0]['local_dir']=str(self.pup)
        self.index={'_class_name':'Flux3DistilledModularPipeline','_blocks_class_name':'Flux3DistilledAutoBlocks'}
        classes={'transformer':('diffusers','Flux3Transformer2DModel'),'vae':('diffusers','AutoencoderKLFlux2'),
                 'text_encoder':('transformers','Qwen3VLForConditionalGeneration'),'tokenizer':('transformers','Qwen3VLProcessor'),
                 'prompt_upsampler':('transformers','Qwen3_5ForConditionalGeneration'),
                 'prompt_upsampler_processor':('transformers','AutoProcessor')}
        for name,(library,cls) in classes.items():
            folder=adapter.COMPONENT_FOLDERS.get(name,'')
            self.index[name]=[library,cls,{'pretrained_model_name_or_path':'REMOTE/NEVER-LOAD',
                'subfolder':folder,'revision':'remote-commit','type_hint':[library,cls]}]
        (self.model/'modular_model_index.json').write_text(json.dumps(self.index))
        for name in ('transformer','vae','text_encoder'):
            folder=self.model/name;folder.mkdir();(folder/'config.json').write_text('{}');(folder/'model.safetensors').write_bytes(b'fake')
        for name in ('config.json','tokenizer_config.json','processor_config.json'):(self.pup/name).write_text('{}')
        (self.pup/'system_prompts.json').write_text(json.dumps({'i2i':'official fixture prompt'}))
        (self.pup/'model.safetensors').write_bytes(b'official-pup-fixture')
        self.loaded=[];self.moves=[];self.created=[]

    def tearDown(self):self.temp.cleanup()

    def fake_modules(self,*,missing=(),with_pup=True):
        outer=self
        class Pipe:
            vae_scale_factor=8
            null_component_names=list(missing)
            def __init__(self,**kwargs):
                outer.created.append(kwargs)
                self.blocks=SimpleNamespace(sub_blocks={'prompt_upsampler':object()} if with_pup else {})
            def load_components(self,**kwargs):outer.loaded.append(kwargs)
            def to(self,device):outer.moves.append(device)
        class Manager:
            def enable_auto_cpu_offload(self,**kwargs):outer.moves.append(kwargs)
        module=ModuleType('diffusers');module.Flux3DistilledModularPipeline=Pipe;module.ComponentsManager=Manager
        prompts=ModuleType('diffusers.modular_pipelines.flux3.prompt_upsampler');prompts.SYSTEM_PROMPTS={'i2i':'official fixture prompt'}
        return {'diffusers':module,'diffusers.modular_pipelines.flux3.prompt_upsampler':prompts}

    def test_validate_modular_components_and_local_pup(self):
        self.assertEqual(validate_model_files(self.model,self.config),self.index)
        (self.pup/'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'PUP weights'):validate_model_files(self.model,self.config)

    def test_all_specs_are_local_and_downloaded_index_is_immutable(self):
        before=(self.model/'modular_model_index.json').read_bytes()
        localized=adapter.local_modular_index(self.model,self.config)
        for name in (*adapter.COMPONENT_FOLDERS,*adapter.PUP_COMPONENTS):
            spec=localized[name][2]
            self.assertEqual(spec['pretrained_model_name_or_path'],str(self.pup if name in adapter.PUP_COMPONENTS else self.model))
            self.assertIsNone(spec['revision'])
        self.assertEqual((self.model/'modular_model_index.json').read_bytes(),before)

    def test_load_is_offline_float32_vae_and_exactly_one_builtin_pup(self):
        torch=SimpleNamespace(float32='float32',bfloat16='bfloat16')
        with patch.dict(sys.modules,self.fake_modules()):pipe=adapter.load_flux3_pipeline(self.config,self.model,torch,'cuda:0',False)
        self.assertEqual(self.loaded,[{'dtype':{'vae':'float32','default':'bfloat16'},'local_files_only':True}])
        self.assertEqual(self.moves,['cuda:0'])
        self.assertNotIn('pretrained_model_name_or_path',self.created[0])
        self.assertIn('prompt_upsampler',pipe.blocks.sub_blocks)

    def test_missing_pup_or_failed_component_cannot_silently_generate(self):
        torch=SimpleNamespace(float32='float32',bfloat16='bfloat16')
        with patch.dict(sys.modules,self.fake_modules(with_pup=False)),self.assertRaisesRegex(ValueError,'PUP block missing'):
            adapter.load_flux3_pipeline(self.config,self.model,torch,'cuda',False)
        with patch.dict(sys.modules,self.fake_modules(missing=['prompt_upsampler'])),self.assertRaisesRegex(ValueError,'failed to load'):
            adapter.load_flux3_pipeline(self.config,self.model,torch,'cuda',False)

    def test_modular_output_keeps_original_and_expanded_instructions_separate(self):
        image=Image.new('RGB',(64,64),'white');original="Delete 'SALE'.";expanded='Remove the SALE lettering from the supplied poster.'
        kwargs=inference_kwargs(self.config,instruction=original,image=image,width=64,height=64,generator='seed42')
        self.assertEqual(kwargs['prompt'],original);self.assertIs(kwargs['image'],image)
        self.assertEqual(kwargs['num_inference_steps'],16);self.assertEqual(kwargs['output'],['images','prompt'])
        self.assertNotIn('guidance_scale',kwargs);self.assertNotIn('boxes',kwargs)
        returned,metadata=extract_generation_output(self.config,{'images':[image],'prompt':expanded},original)
        self.assertIs(returned,image);self.assertEqual(metadata['original_instruction'],original)
        self.assertEqual(metadata['model_prompt'],expanded);self.assertEqual(metadata['prompt_preprocessing']['passes'],1)
        with self.assertRaisesRegex(ValueError,'nonempty caption'):extract_generation_output(self.config,{'images':[image],'prompt':''},original)

    def test_qwen21_components_and_turbo_checkpoint_schedule_are_honored(self):
        turbo=self.root/'turbo';turbo.mkdir();config=get_model_config('qwen-image21-turbo')
        index={'_class_name':'QwenImage21Pipeline','sample_sigmas':[1,.978453,.95418,.926626,.89508,.845148,.704534,.414568]}
        for name in config['required_components']:
            directory=turbo/name;directory.mkdir();(directory/'config.json').write_text('{}');index[name]=['fixture','Component']
            if name in {'transformer','vae','text_encoder'} or name.startswith('text_encoder_'):
                (directory/'model.safetensors').write_bytes(b'fixture checkpoint bytes')
        (turbo/'model_index.json').write_text(json.dumps(index))
        validate_model_files(turbo,config)
        self.assertEqual(config['sample_sigmas'],index['sample_sigmas'])
        self.assertNotIn('tokenizer',config['required_components'])
        kwargs=inference_kwargs(config,instruction="Delete 'A'.",image='source',width=64,height=64,generator='seed42')
        self.assertEqual(kwargs['num_inference_steps'],8);self.assertTrue(kwargs['use_kv_cache'])
        self.assertNotIn('sigmas',kwargs)  # Let the pipeline consume its saved schedule.
        config['num_inference_steps']=40
        with self.assertRaisesRegex(ValueError,'cannot silently override'):validate_model_files(turbo,config)


if __name__=='__main__':unittest.main()
