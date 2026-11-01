"""Native SenseNova input, output and local-checkpoint contracts; no GPU/API."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from sensenova_adapter import SenseNovaEditingPipeline, validate_sensenova_files, validate_sensenova_source
from model_registry import get_model_config
from download_models import select_diffusers_files


class SenseNovaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=get_model_config('sensenova-u1.5-8b-mot')
        (self.root/'config.json').write_text(json.dumps({'architectures':['NEOChatModel'],
            'model_type':'neo_chat','patch_size':16,'downsample_ratio':0.5}))
        for name in ['tokenizer_config.json','vocab.json','merges.txt']:(self.root/name).write_text('{}')
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer':'part.safetensors'}}))
        (self.root/'part.safetensors').write_bytes(b'fixture')

    def test_native_checkpoint_and_every_referenced_shard_are_required(self):
        self.assertEqual(validate_sensenova_files(self.root,self.config)['model_type'],'neo_chat')
        (self.root/'part.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'Missing SenseNova checkpoint shard'):
            validate_sensenova_files(self.root,self.config)

    def test_unsafe_shards_and_wrong_architecture_are_rejected(self):
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer':'../part.safetensors'}}))
        with self.assertRaisesRegex(ValueError,'Unsafe SenseNova shard'):validate_sensenova_files(self.root,self.config)
        (self.root/'config.json').write_text('{"model_type":"another-model"}')
        with self.assertRaisesRegex(ValueError,'native NEOChatModel'):validate_sensenova_files(self.root,self.config)

    def test_native_source_signature_is_checked_without_importing_model(self):
        path=self.root/'models/neo_unify/modeling_neo_chat.py';path.parent.mkdir(parents=True)
        signature='self,tokenizer,prompt,images,image_size,cfg_scale,img_cfg_scale,cfg_norm,timestep_shift,cfg_interval,num_steps,batch_size,think_mode,seed'
        path.write_text(f'class NEOChatModel:\n def it2i_generate({signature}): pass\n')
        self.assertIn('image',validate_sensenova_source(self.root))
        path.write_text('class NEOChatModel: pass\n')
        with self.assertRaisesRegex(ValueError,'signature is incompatible'):validate_sensenova_source(self.root)

    def test_one_original_source_instruction_seed_and_official_tensor_conversion(self):
        import torch
        tensor=torch.linspace(-1.2,1.2,1*3*32*64).reshape(1,3,32,64).to(torch.bfloat16)
        model=SimpleNamespace(it2i_generate=Mock(return_value=tensor))
        pipe=SenseNovaEditingPipeline(model,'tokenizer')
        image=Image.new('RGB',(64,32),'blue');generator=torch.Generator().manual_seed(42)
        result=pipe(prompt="Delete 'a'.",image=image,width=64,height=32,num_inference_steps=50,
                    generator=generator,**self.config['call_kwargs'])
        args,kw=model.it2i_generate.call_args
        self.assertEqual(args[:2],('tokenizer',"Delete 'a'."));self.assertEqual(len(args[2]),1)
        self.assertIs(args[2][0],image)
        self.assertEqual(kw,{'image_size':(64,32),'num_steps':50,'cfg_scale':4.,'img_cfg_scale':1.,
            'cfg_norm':'none','timestep_shift':3.,'cfg_interval':(0.,1.),'batch_size':1,'think_mode':False,'seed':42})
        mean=torch.tensor((.5,.5,.5)).view(1,3,1,1);std=mean.clone()
        expected=((tensor.float()*std+mean).clamp(0,1).permute(0,2,3,1).numpy()*255).round().astype(np.uint8)[0]
        np.testing.assert_array_equal(np.asarray(result.images[0]),expected)
        self.assertEqual(result.images[0].mode,'RGB');self.assertEqual(result.images[0].size,image.size)

    def test_native_download_keeps_root_weights_and_tokenizer(self):
        names=[p.name for p in self.root.iterdir()]+['README.md','.gitattributes']
        info=SimpleNamespace(siblings=[SimpleNamespace(rfilename=name,size=10) for name in names])
        selected={r['path'] for r in select_diffusers_files(info,self.config)}
        self.assertIn('part.safetensors',selected);self.assertIn('vocab.json',selected)
        self.assertNotIn('.gitattributes',selected)


if __name__=='__main__':unittest.main()
