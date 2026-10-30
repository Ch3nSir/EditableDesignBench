"""Offline, single-PUP adapter for the official FLUX3 distilled modular model."""
from copy import deepcopy
import ast
import json
from pathlib import Path

from generation_provenance import content_tree,_package_root,canonical_digest,PROVENANCE_VERSION

COMPONENT_FOLDERS={'transformer':'transformer','vae':'vae','text_encoder':'text_encoder','tokenizer':'text_encoder'}
PUP_COMPONENTS=('prompt_upsampler','prompt_upsampler_processor')


def pup_directory(config):
    dependencies=config.get('dependencies')
    if not isinstance(dependencies,list) or len(dependencies)!=1 or dependencies[0].get('role')!='prompt_upsampler':
        raise ValueError('FLUX3 requires exactly one pinned local official PUP dependency')
    return Path(dependencies[0]['local_dir']).resolve()


def _has_weights(folder):
    return any(p.is_file() for pattern in ('*.safetensors','*.bin') for p in folder.glob(pattern))


def _check_shards(folder):
    for index in folder.glob('*.index.json'):
        mapping=json.loads(index.read_text()).get('weight_map',{})
        for filename in mapping.values():
            path=Path(filename)
            if path.is_absolute() or '..' in path.parts or not (folder/path).is_file():
                raise ValueError('Missing or unsafe checkpoint shard: '+str(filename))


def validate_flux3_files(model_path,config):
    root=Path(model_path).resolve();index=json.loads((root/'modular_model_index.json').read_text())
    if index.get('_class_name')!='Flux3DistilledModularPipeline' or index.get('_blocks_class_name')!='Flux3DistilledAutoBlocks':
        raise ValueError('FLUX3 requires the official distilled16 modular checkpoint, with its automatic PUP block')
    for name,folder_name in COMPONENT_FOLDERS.items():
        component=index.get(name)
        if not isinstance(component,list) or len(component)!=3 or component[2].get('subfolder')!=folder_name:
            raise ValueError('Unexpected FLUX3 modular component: '+name)
        folder=root/folder_name
        if not folder.is_dir() or not any(folder.iterdir()):raise ValueError('Missing FLUX3 component '+folder_name)
        if name!='tokenizer' and not _has_weights(folder):raise ValueError('Missing FLUX3 component weights '+folder_name)
        _check_shards(folder)
    pup=pup_directory(config)
    for name in PUP_COMPONENTS:
        component=index.get(name)
        if not isinstance(component,list) or len(component)!=3:raise ValueError('Missing official PUP component '+name)
    for filename in ('config.json','system_prompts.json','tokenizer_config.json','processor_config.json'):
        if not (pup/filename).is_file():raise ValueError('Missing local PUP file '+filename)
    if not _has_weights(pup):raise ValueError('Missing local PUP weights')
    _check_shards(pup)
    prompts=json.loads((pup/'system_prompts.json').read_text())
    if not isinstance(prompts.get('i2i'),str) or not prompts['i2i'].strip():raise ValueError('Local PUP lacks its official image-editing system prompt')
    return index


def local_modular_index(model_path,config):
    """Rewrite loading specs in memory; never alter downloaded checkpoint files."""
    root=Path(model_path).resolve();index=deepcopy(json.loads((root/'modular_model_index.json').read_text()))
    pup=pup_directory(config)
    for name in (*COMPONENT_FOLDERS,*PUP_COMPONENTS):
        spec=index[name][2]
        spec['pretrained_model_name_or_path']=str(pup if name in PUP_COMPONENTS else root)
        spec['subfolder']='' if name in PUP_COMPONENTS else COMPONENT_FOLDERS[name]
        spec['revision']=None
    return index


def validate_flux3_source(package_root):
    root=Path(package_root);folder=root/'modular_pipelines/flux3'
    required=('modular_pipeline.py','modular_blocks_flux3_distilled.py','prompt_upsampler.py')
    for name in required:
        if not (folder/name).is_file():raise ValueError('Installed Diffusers lacks current FLUX3 modular/PUP support; install the pinned benchmark wheel')
    pipeline=(folder/'modular_pipeline.py').read_text();blocks=(folder/'modular_blocks_flux3_distilled.py').read_text();pup=(folder/'prompt_upsampler.py').read_text()
    tree=ast.parse(pipeline)
    cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Flux3DistilledModularPipeline'),None)
    if cls is None or 'Flux3DistilledAutoBlocks' not in pipeline or 'Flux3AutoPromptUpsampleStep' not in blocks or '"prompt_upsampler"' not in blocks:
        raise ValueError('FLUX3 default blocks must include exactly one official prompt-upsample stage')
    if 'do_sample=False' not in pup or 'enable_thinking=False' not in pup or 'prepare_references' not in pup:
        raise ValueError('Cannot verify official deterministic source-conditioned PUP implementation')
    return ['generator','height','image','max_new_tokens','num_inference_steps','output','prompt','width']


def load_flux3_pipeline(config,model_path,torch,device,cpu_offload):
    import diffusers
    from diffusers.modular_pipelines.flux3.prompt_upsampler import SYSTEM_PROMPTS
    official=json.loads((pup_directory(config)/'system_prompts.json').read_text())
    if SYSTEM_PROMPTS.get('i2i')!=official.get('i2i'):
        raise ValueError('Installed FLUX3 PUP system prompt differs from the pinned official dependency')
    index=local_modular_index(model_path,config)
    manager=diffusers.ComponentsManager() if cpu_offload else None
    # Construct from the already-localized public config. Passing the image
    # directory again would let generic local-repo resolution rebind external
    # root PUP specs to image weights when flat files also happen to exist.
    pipe=diffusers.Flux3DistilledModularPipeline(modular_config_dict=index,components_manager=manager)
    if 'prompt_upsampler' not in pipe.blocks.sub_blocks:
        raise ValueError('FLUX3 built-in PUP block missing; refusing short-instruction inference')
    pipe.load_components(dtype={'vae':torch.float32,'default':torch.bfloat16},local_files_only=True)
    missing=set(pipe.null_component_names)&set((*COMPONENT_FOLDERS,*PUP_COMPONENTS))
    if missing:raise ValueError('FLUX3 local components failed to load: '+', '.join(sorted(missing)))
    # Verify that the installed official PUP prompt text matches the downloaded
    # dependency whose content is pinned in the generation protocol.
    if cpu_offload:manager.enable_auto_cpu_offload(device=device)
    else:pipe.to(device)
    return pipe


def collect_flux3_content(model_path,config):
    """Pin converted local checkpoint + official local PUP + actual source trees."""
    from importlib import metadata
    packages={}
    for name in ('diffusers','transformers'):
        packages[name]={**content_tree(_package_root(name),python_only=True),'version':metadata.version(name),
                        'scope':'All resolved package .py source files'}
    value={'version':PROVENANCE_VERSION,'model':content_tree(model_path),
        'dependencies':{'prompt_upsampler':content_tree(pup_directory(config))},'python_packages':packages,
        'hash_policy':'Fresh streaming SHA-256 of model, local PUP and Python source bytes; no timestamp cache'}
    return {**value,'sha256':canonical_digest(value)}
