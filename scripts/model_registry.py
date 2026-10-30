"""Explicit adapters for the supported local image-editing checkpoints.

Importing this registry does not import torch or load models. Each invocation
receives exactly one case image and its original instruction.
"""
from copy import deepcopy
import ast
from importlib.util import find_spec
import json
import math
from pathlib import Path

from settings import MODEL_ROOT
GENERATION_PROTOCOL_VERSION='multi-image-edit-generation-v1'
RESOLUTION_PROTOCOL='native-pad32-v1'
DEFAULT_MODELS=('flux3-klein','qwen-image21','qwen-image21-turbo')

MODEL_CONFIGS={
    'flux3-klein':{
        'repo_id':'black-forest-labs/flux-3-image-klein',
        'local_dir':str(MODEL_ROOT/'flux-3-image-klein'),
        'backend':'flux3-modular-local-pup',
        'pipeline':'Flux3DistilledModularPipeline',
        'pipeline_source':'flux3/modular_pipeline.py',
        'num_inference_steps':16,'padding_multiple':32,
        'call_kwargs':{'output':['images','prompt'],'max_new_tokens':3584},
        'cpu_offload':False,
        'required_components':['transformer','vae','text_encoder'],
        'dependencies':[{'id':'flux3-klein-pup','repo_id':'black-forest-labs/flux-3-image-klein-pup',
            'local_dir':str(MODEL_ROOT/'flux-3-image-klein-pup'),'role':'prompt_upsampler'}],
        'conditioning_policy':'Official distilled16 modular pipeline; exactly one source; exactly one built-in local official PUP pass. Original instruction remains the scoring instruction.',
    },
    'qwen-image21':{
        'repo_id':'Qwen/Qwen-Image-2.1','local_dir':str(MODEL_ROOT/'Qwen-Image-2.1'),
        'backend':'diffusion-pipeline','pipeline':'QwenImage21Pipeline',
        'pipeline_source':'qwenimage21/pipeline_qwenimage21.py',
        'num_inference_steps':40,'padding_multiple':32,
        'call_kwargs':{'true_cfg_scale':1.0,'use_kv_cache':True,'output_resolution':2048},
        'cpu_offload':False,'prefer_existing':True,
        'required_components':['processor','scheduler','text_encoder','transformer','vae'],
        'conditioning_policy':'Official Qwen21 reference preprocessing and KV cache; one source and original instruction.',
    },
    'qwen-image21-turbo':{
        'repo_id':'Qwen/Qwen-Image-2.1-Turbo','local_dir':str(MODEL_ROOT/'Qwen-Image-2.1-Turbo'),
        'backend':'diffusion-pipeline','pipeline':'QwenImage21Pipeline',
        'pipeline_source':'qwenimage21/pipeline_qwenimage21.py',
        'num_inference_steps':8,'padding_multiple':32,
        'call_kwargs':{'true_cfg_scale':1.0,'use_kv_cache':True,'output_resolution':2048},
        'cpu_offload':False,'checkpoint_sample_sigmas':True,
        'required_components':['processor','scheduler','text_encoder','transformer','vae'],
        'conditioning_policy':'Official Qwen21 Turbo checkpoint sample_sigmas, CFG1 and KV cache; one source and original instruction.',
    },
    'qwen-edit-2511':{
        'repo_id':'Qwen/Qwen-Image-Edit-2511',
        'local_dir':str(MODEL_ROOT/'Qwen-Image-Edit-2511'),
        'pipeline':'QwenImageEditPlusPipeline',
        'pipeline_source':'qwenimage/pipeline_qwenimage_edit_plus.py',
        'num_inference_steps':40,
        'padding_multiple':32,
        'call_kwargs':{'true_cfg_scale':4.0,'negative_prompt':' ','guidance_scale':1.0},
        'cpu_offload':False,
        'required_components':['transformer','vae','scheduler','tokenizer','text_encoder','processor'],
        'conditioning_policy':'Official Qwen Plus internal conditioning preprocessing, one source image.',
    },
    'flux2-klein-9b':{
        'repo_id':'black-forest-labs/FLUX.2-klein-9B',
        'local_dir':str(MODEL_ROOT/'FLUX.2-klein-9B'),
        'pipeline':'Flux2KleinPipeline',
        'pipeline_source':'flux2/pipeline_flux2_klein.py',
        'num_inference_steps':4,
        'padding_multiple':32,
        'call_kwargs':{'guidance_scale':1.0},
        'cpu_offload':False,
        'required_components':['transformer','vae','scheduler','tokenizer','text_encoder'],
        'is_distilled':True,
        'conditioning_policy':'Official FLUX Klein conditioning preprocessing; one source image, no additional references.',
    },
    'flux2-dev':{
        'repo_id':'black-forest-labs/FLUX.2-dev',
        'local_dir':str(MODEL_ROOT/'FLUX.2-dev'),
        'pipeline':'Flux2Pipeline',
        'pipeline_source':'flux2/pipeline_flux2.py',
        'num_inference_steps':50,
        'padding_multiple':32,
        'call_kwargs':{'guidance_scale':4.0,'caption_upsample_temperature':None},
        'cpu_offload':True,
        'required_components':['transformer','vae','scheduler','tokenizer','text_encoder'],
        'conditioning_policy':'Official FLUX2 conditioning preprocessing; caption upsampling disabled, original instruction.',
    },
}


def get_model_config(model_id):
    if model_id not in MODEL_CONFIGS:raise ValueError('Unsupported model ID: '+str(model_id))
    return deepcopy(MODEL_CONFIGS[model_id])


def _validate_component_checkpoint(folder):
    """Require local checkpoint bytes, including every indexed shard."""
    folder=Path(folder)
    weights=[path for path in folder.iterdir()
             if path.is_file() and path.suffix in {'.safetensors','.bin'} and path.stat().st_size>0]
    if not weights:
        raise ValueError('Model component checkpoint weights missing: '+folder.name)
    for path in folder.glob('*.index.json'):
        index=json.loads(path.read_text())
        mapping=index.get('weight_map') if isinstance(index,dict) else None
        if not isinstance(mapping,dict) or not mapping:
            raise ValueError('Invalid checkpoint weight_map: '+folder.name)
        for shard in mapping.values():
            if not isinstance(shard,str):raise ValueError('Invalid checkpoint shard path: '+folder.name)
            relative=Path(shard)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe checkpoint shard path: '+folder.name)
            target=folder/relative
            if not target.is_file() or target.stat().st_size<=0:
                raise ValueError('Model component checkpoint shard missing: '+folder.name+'/'+shard)


def validate_model_files(model_path,config):
    """Check local pipeline identity/components before any GPU allocation/load."""
    model_path=Path(model_path)
    if config.get('backend')=='flux3-modular-local-pup':
        from flux3_adapter import validate_flux3_files
        return validate_flux3_files(model_path,config)
    index_path=model_path/'model_index.json'
    index=json.loads(index_path.read_text())
    if index.get('_class_name')!=config['pipeline']:
        raise ValueError(f'Model pipeline mismatch: expected {config["pipeline"]}, got {index.get("_class_name")}')
    for name in config['required_components']:
        component=index.get(name)
        if not isinstance(component,list) or len(component)!=2 or not all(isinstance(v,str) and v for v in component):
            raise ValueError(f'Missing/invalid model component in model_index.json: {name}')
        if not (model_path/name).is_dir() or not any(p.is_file() for p in (model_path/name).iterdir()):
            raise ValueError(f'Model component files missing: {name}')
        if name in {'transformer','vae','text_encoder'} or name.startswith('text_encoder_'):
            _validate_component_checkpoint(model_path/name)
    if 'is_distilled' in config and index.get('is_distilled') is not config['is_distilled']:
        raise ValueError('FLUX Klein checkpoint must explicitly declare is_distilled=true; base and distilled weights cannot be mixed')
    if config.get('checkpoint_sample_sigmas'):
        sigmas=index.get('sample_sigmas')
        if (not isinstance(sigmas,list) or len(sigmas)!=8 or
                any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=1 for v in sigmas)
                or any(a<=b for a,b in zip(sigmas,sigmas[1:]))):
            raise ValueError('Qwen21 Turbo requires its eight descending checkpoint sample_sigmas')
        if config['num_inference_steps']!=len(sigmas):
            raise ValueError('Turbo uses eight checkpoint sample_sigmas; --steps cannot silently override that schedule')
        config['sample_sigmas']=list(sigmas)
    return index


def validate_pipeline_signature(config):
    """Check installed call support from source without importing torch."""
    spec=find_spec('diffusers')
    locations=list(spec.submodule_search_locations or []) if spec else []
    if len(locations)!=1:raise ValueError('Cannot resolve installed Diffusers source')
    if config.get('backend')=='flux3-modular-local-pup':
        from flux3_adapter import validate_flux3_source
        return validate_flux3_source(Path(locations[0]))
    path=Path(locations[0])/'pipelines'/config['pipeline_source']
    if not path.is_file():raise ValueError('Installed Diffusers lacks '+config['pipeline'])
    tree=ast.parse(path.read_text())
    cls=next((node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==config['pipeline']),None)
    function=next((node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='__call__'),None) if cls else None
    if function is None:raise ValueError('Cannot verify pipeline call signature')
    accepted={a.arg for a in (*function.args.args,*function.args.kwonlyargs)}
    requested={'prompt','image','width','height','num_inference_steps','generator',*config['call_kwargs']}
    if not requested<=accepted:raise ValueError('Installed pipeline does not accept: '+', '.join(sorted(requested-accepted)))
    if config.get('checkpoint_sample_sigmas'):
        if 'sample_sigmas' not in path.read_text():raise ValueError('Installed Qwen21 pipeline lacks checkpoint sample_sigmas support')
    return sorted(requested)


def inference_kwargs(config,*,instruction,image,width,height,generator):
    """Model-specific calls, with a single source image rather than a batch."""
    return {'prompt':instruction,'image':image,'width':width,'height':height,
            'num_inference_steps':config['num_inference_steps'],'generator':generator,
            **deepcopy(config['call_kwargs'])}


def load_pipeline(config,model_path,torch,device,cpu_offload):
    if config.get('backend')=='flux3-modular-local-pup':
        from flux3_adapter import load_flux3_pipeline
        return load_flux3_pipeline(config,model_path,torch,device,cpu_offload)
    import diffusers
    pipeline_class=getattr(diffusers,config['pipeline'])
    pipe=pipeline_class.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,local_files_only=True)
    if cpu_offload:pipe.enable_model_cpu_offload(device=device)
    else:pipe.to(device)
    return pipe


def extract_generation_output(config,result,instruction):
    """Normalize API return types while retaining the official model prompt."""
    if config.get('backend')=='flux3-modular-local-pup':
        if not isinstance(result,dict) or not isinstance(result.get('prompt'),str) or not result['prompt'].strip():
            raise ValueError('FLUX3 modular output must include the nonempty caption from its single built-in PUP pass')
        images=result.get('images')
        if not isinstance(images,(list,tuple)) or len(images)!=1:raise ValueError('Expected one FLUX3 image')
        return images[0],{'original_instruction':instruction,'model_prompt':result['prompt'],
            'prompt_preprocessing':{'method':'official-built-in-local-flux3-pup','passes':1,
                'repo_id':config['dependencies'][0]['repo_id'],'do_sample':False,'enable_thinking':False,
                'max_new_tokens':config['call_kwargs']['max_new_tokens'],
                'input_scope':'Only the current source image and original instruction; no target, mask or other cases.'}}
    images=result.images
    if len(images)!=1:raise ValueError('Expected one generated image')
    return images[0],{'original_instruction':instruction,'model_prompt':instruction,
                      'prompt_preprocessing':{'method':'none','passes':0}}
