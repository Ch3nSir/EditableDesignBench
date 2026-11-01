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
import re

from settings import MODEL_ROOT
GENERATION_PROTOCOL_VERSION='multi-image-edit-generation-v1'
RESOLUTION_PROTOCOL='native-pad32-v1'
DEFAULT_MODELS=('flux3-klein','qwen-image21','qwen-image21-turbo')

MODEL_CONFIGS={
    'fibo2-edit-turbo':{
        'repo_id':'briaai/Fibo-2.0-edit-turbo','local_dir':str(MODEL_ROOT/'Fibo-2.0-edit-turbo'),
        'backend':'diffusion-pipeline','pipeline':'BriaFibo2Pipeline','pipeline_source':'bria_fibo2/pipeline_bria_fibo2.py',
        'required_components':['text_encoder','processor','scheduler','transformer','vae'],
        'num_inference_steps':4,'padding_multiple':32,'cpu_offload':False,
        'call_kwargs':{'guidance_scale':1.0},
        'conditioning_policy':'Official four-step no-guidance Diffusers direct editing; one original source and instruction; no optional VGL/Gemini rewrite or mask; explicit native output.',
    },
    'joyai-image-edit-plus':{
        'repo_id':'jdopensource/JoyAI-Image-Edit-Plus-Diffusers',
        'local_dir':str(MODEL_ROOT/'JoyAI-Image-Edit-Plus-Diffusers'),
        'backend':'diffusion-pipeline','pipeline':'JoyImageEditPlusPipeline',
        'pipeline_source':'joyimage/pipeline_joyimage_edit_plus.py',
        'image_argument':'images','vae_scale_factor_attribute':'vae_scale_factor_spatial',
        'required_components':['text_encoder','tokenizer','processor','scheduler','transformer','vae'],
        'num_inference_steps':30,'padding_multiple':32,'cpu_offload':False,
        'call_kwargs':{'guidance_scale':4.0,'negative_prompt':'low quality, blurry, deformed'},
        'conditioning_policy':'Official Plus model at 30 steps and CFG4; images receives exactly one original source; original instruction and explicit native output dimensions. Official internal reference bucketing retained; no prompt rewrite, masks or extra references.',
    },
    'sensenova-u1.5-8b-mot':{
        'repo_id':'sensenova/SenseNova-U1.5-8B-MoT',
        'local_dir':str(MODEL_ROOT/'SenseNova-U1.5-8B-MoT'),
        'backend':'sensenova-native','pipeline':'SenseNovaEditingPipeline',
        'num_inference_steps':50,'padding_multiple':32,
        'call_kwargs':{'cfg_scale':4.0,'img_cfg_scale':1.0,'cfg_norm':'none',
                       'timestep_shift':3.0,'cfg_interval':[0.0,1.0]},
        'cpu_offload':False,'prefer_existing':True,'attn_backend':'sdpa',
        'conditioning_policy':'Official native it2i_generate; one source and original instruction; explicit pad32 output size, native internal input preprocessing; no external resizing, Edit PE or think mode.',
    },
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


def load_model_config(path):
    """Register extra Diffusers models without modifying built-in profiles.

    The JSON file contains public generation settings, never API credentials.
    Validate the entire file before updating the process-local registry.
    """
    if path is None:return
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict) or set(data)!={'models'} or not isinstance(data['models'],dict) or not data['models']:
        raise ValueError('Model config requires a nonempty models object')
    pending={}
    allowed={'repo_id','local_dir','pipeline','pipeline_source','num_inference_steps',
             'padding_multiple','call_kwargs','cpu_offload','required_components',
             'is_distilled','checkpoint_sample_sigmas','conditioning_policy'}
    reserved={'prompt','image','images','width','height','num_inference_steps','generator',
              'mask','mask_image','target','target_image','reference_image','reference_images','control_image'}
    for model_id,item in data['models'].items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',model_id) or model_id=='all':
            raise ValueError('Invalid custom model ID')
        if not isinstance(item,dict) or set(item)-allowed:
            raise ValueError('Unsupported custom model fields: '+model_id)
        config=deepcopy(item)
        for name in ('repo_id','pipeline','pipeline_source','num_inference_steps','required_components'):
            if name not in config:raise ValueError('Missing custom model field: '+name)
        if not isinstance(config['repo_id'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',config['repo_id']):
            raise ValueError('Custom repo_id must be owner/repository')
        if not isinstance(config['pipeline'],str) or not config['pipeline'].isidentifier():
            raise ValueError('Custom pipeline must name an installed Diffusers class')
        if not isinstance(config['pipeline_source'],str):raise ValueError('pipeline_source must be a relative Python source path')
        source=Path(config['pipeline_source'])
        if source.is_absolute() or '..' in source.parts or source.suffix!='.py':
            raise ValueError('pipeline_source must be relative to diffusers/pipelines')
        steps=config['num_inference_steps']
        if type(steps) is not int or steps<1:raise ValueError('Custom steps must be a positive integer')
        components=config['required_components']
        if (not isinstance(components,list) or not components or
                any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_]+',v) for v in components)
                or len(set(components))!=len(components)):
            raise ValueError('Custom required_components must list distinct component folder names')
        config.setdefault('call_kwargs',{})
        if not isinstance(config['call_kwargs'],dict) or reserved & set(config['call_kwargs']):
            raise ValueError('Custom call_kwargs cannot override source, instruction, size, seed or steps')
        # JSON's permissive NaN/Infinity values must never enter a run protocol.
        try:json.dumps(config,allow_nan=False)
        except ValueError:raise ValueError('Custom config must contain finite JSON values') from None
        for key in ('cpu_offload','is_distilled','checkpoint_sample_sigmas'):
            if key in config and type(config[key]) is not bool:raise ValueError(key+' must be boolean')
        config.setdefault('cpu_offload',False)
        if config.setdefault('padding_multiple',32)!=32:raise ValueError('Custom generation currently requires pad32 geometry')
        local=config.get('local_dir',model_id)
        if not isinstance(local,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',local):
            raise ValueError('Custom local_dir must be a folder name under the model root')
        config['local_dir']=str(MODEL_ROOT/local)
        config['backend']='diffusion-pipeline'
        config.setdefault('conditioning_policy','Single source image and original instruction; no target or masks.')
        if model_id in MODEL_CONFIGS and MODEL_CONFIGS[model_id]!=config:
            raise ValueError('Cannot replace an existing model profile: '+model_id)
        occupied={**MODEL_CONFIGS,**pending}
        if (any(other!=model_id and old['local_dir']==config['local_dir'] for other,old in occupied.items())
                or any(dep['local_dir']==config['local_dir'] for old in occupied.values() for dep in old.get('dependencies',[]))):
            raise ValueError('Custom models must use distinct checkpoint directories')
        pending[model_id]=config
    MODEL_CONFIGS.update(pending)


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
    if config.get('backend')=='sensenova-native':
        from sensenova_adapter import validate_sensenova_files
        return validate_sensenova_files(model_path,config)
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
        if name in {'transformer','unet','vae','text_encoder','image_encoder'} or name.startswith('text_encoder_'):
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
    if config.get('backend')=='sensenova-native':
        from sensenova_adapter import validate_sensenova_source
        return validate_sensenova_source()
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
    requested={'prompt',config.get('image_argument','image'),'width','height','num_inference_steps','generator',*config['call_kwargs']}
    if not requested<=accepted:raise ValueError('Installed pipeline does not accept: '+', '.join(sorted(requested-accepted)))
    if config.get('checkpoint_sample_sigmas'):
        if 'sample_sigmas' not in path.read_text():raise ValueError('Installed Qwen21 pipeline lacks checkpoint sample_sigmas support')
    return sorted(requested)


def inference_kwargs(config,*,instruction,image,width,height,generator):
    """Model-specific calls, with a single source image rather than a batch."""
    image_argument=config.get('image_argument','image')
    return {'prompt':instruction,image_argument:[image] if image_argument=='images' else image,'width':width,'height':height,
            'num_inference_steps':config['num_inference_steps'],'generator':generator,
            **deepcopy(config['call_kwargs'])}


def load_pipeline(config,model_path,torch,device,cpu_offload):
    if config.get('backend')=='sensenova-native':
        from sensenova_adapter import load_sensenova_pipeline
        return load_sensenova_pipeline(config,model_path,torch,device,cpu_offload)
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
