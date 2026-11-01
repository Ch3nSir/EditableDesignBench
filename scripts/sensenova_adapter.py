"""Single-source native SenseNova editing with the official sampling/conversion.

This module stays importable without the optional SenseNova package or torch.
The package and checkpoint are resolved and pinned before model loading.
"""
import ast
from importlib import metadata
import json
from pathlib import Path
from types import SimpleNamespace

from generation_provenance import content_tree, _package_root, canonical_digest, PROVENANCE_VERSION


def validate_sensenova_files(model_path, config):
    root=Path(model_path)
    index=json.loads((root/'config.json').read_text())
    if index.get('model_type')!='neo_chat' or index.get('architectures')!=['NEOChatModel']:
        raise ValueError('SenseNova requires its native NEOChatModel checkpoint')
    if index.get('patch_size')!=16 or index.get('downsample_ratio')!=0.5:
        raise ValueError('SenseNova checkpoint must use the supported 32-pixel image grid')
    for name in ('tokenizer_config.json','vocab.json','merges.txt','model.safetensors.index.json'):
        if not (root/name).is_file():raise ValueError('Missing SenseNova file: '+name)
    mapping=json.loads((root/'model.safetensors.index.json').read_text()).get('weight_map')
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Invalid SenseNova weight map')
    for shard in mapping.values():
        path=Path(shard) if isinstance(shard,str) else None
        if path is None or path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe SenseNova shard reference')
        if not (root/path).is_file() or (root/path).stat().st_size<=0:
            raise ValueError('Missing SenseNova checkpoint shard: '+shard)
    return index


def validate_sensenova_source(package_root=None):
    try:root=Path(package_root) if package_root is not None else _package_root('sensenova_u1')
    except (ValueError,ModuleNotFoundError):
        raise ValueError('Install the optional official sensenova-u1 package before generation') from None
    path=root/'models/neo_unify/modeling_neo_chat.py'
    if not path.is_file():raise ValueError('Installed SenseNova lacks native image editing')
    tree=ast.parse(path.read_text())
    cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NEOChatModel'),None)
    method=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='it2i_generate'),None) if cls else None
    accepted={a.arg for a in (*method.args.args,*method.args.kwonlyargs)} if method else set()
    required={'tokenizer','prompt','images','image_size','cfg_scale','img_cfg_scale','cfg_norm',
              'timestep_shift','cfg_interval','num_steps','batch_size','think_mode','seed'}
    if not required<=accepted:raise ValueError('Installed SenseNova editing signature is incompatible')
    return ['generator','height','image','num_inference_steps','prompt','width',
            'cfg_scale','img_cfg_scale','cfg_norm','timestep_shift','cfg_interval']


def collect_sensenova_content(model_path):
    packages={}
    for name,distribution in [('sensenova_u1','sensenova-u1'),('transformers','transformers')]:
        packages[name]={**content_tree(_package_root(name),python_only=True),
                        'version':metadata.version(distribution),'scope':'All resolved package .py source files'}
    value={'version':PROVENANCE_VERSION,'model':content_tree(model_path),'python_packages':packages,
           'hash_policy':'Fresh streaming hashes of actual native checkpoint and inference Python sources'}
    return {**value,'sha256':canonical_digest(value)}


class SenseNovaEditingPipeline:
    """Normalize the native model interface without resizing or prompt rewriting."""
    image_grid_factor=32

    def __init__(self, model, tokenizer):
        self.model=model
        self.tokenizer=tokenizer

    def __call__(self, *, prompt, image, width, height, num_inference_steps,
                 generator, cfg_scale, img_cfg_scale, cfg_norm, timestep_shift, cfg_interval):
        import numpy as np
        from PIL import Image
        if image.mode!='RGB' or image.size!=(width,height) or width%32 or height%32:
            raise ValueError('SenseNova requires one padded RGB image and explicit pad32 output dimensions')
        output=self.model.it2i_generate(
            self.tokenizer,prompt,[image],image_size=(width,height),
            num_steps=num_inference_steps,cfg_scale=cfg_scale,img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,timestep_shift=timestep_shift,cfg_interval=tuple(cfg_interval),
            batch_size=1,think_mode=False,seed=generator.initial_seed())
        if tuple(output.shape)!=(1,3,height,width):raise ValueError('Unexpected SenseNova output tensor shape')
        # Exactly the official editing example's _denorm/_to_pil: float32,
        # x * 0.5 + 0.5, clamp, round to uint8. This is the native RGB return.
        pixels=output.float().mul(0.5).add(0.5).clamp(0,1)
        pixels=pixels.permute(0,2,3,1).cpu().numpy()
        rgb=(pixels*255.0).round().astype(np.uint8)[0]
        return SimpleNamespace(images=[Image.fromarray(rgb)])


def load_sensenova_pipeline(config, model_path, torch, device, cpu_offload):
    if cpu_offload:raise ValueError('SenseNova native adapter currently requires full GPU residency')
    import sensenova_u1
    from sensenova_u1.utils import load_model_and_tokenizer
    sensenova_u1.set_attn_backend(config['attn_backend'])
    model,tokenizer=load_model_and_tokenizer(str(model_path),dtype=torch.bfloat16,device=device)
    return SenseNovaEditingPipeline(model,tokenizer)
