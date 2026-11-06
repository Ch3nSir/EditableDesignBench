"""Preserve raw returns and evaluate an explicitly recorded native-image ROI."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ADAPTER_VERSION = 'raw-rgb-roi-v2'
READABLE_ADAPTER_VERSIONS = {'raw-rgb-roi-v1', ADAPTER_VERSION}
# The RGB/ROI format remains v2: only alpha eligibility changes. Keeping the
# format identity preserves scores for already accepted, identical RGB inputs.
ALPHA_POLICY = ('Alpha is diagnostic only; all alpha values accepted; '
                'RGB channels unchanged, no compositing; raw output retained')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instruction_sha256(task):
    return hashlib.sha256(task['instruction'].encode('utf-8')).hexdigest()


def _pair(value, name):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError(f'{name} must be [width,height]')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in value):
        raise ValueError(f'{name} must contain positive integer dimensions')
    return tuple(value)


def validate_geometry(record, source_size=None):
    native=_pair(record.get('source_size'),'source_size')
    canvas=_pair(record.get('canvas_size'),'canvas_size')
    if source_size is not None and native!=tuple(source_size):
        raise ValueError(f'Source size provenance mismatch: record={native}, actual={tuple(source_size)}')
    if canvas[0]<native[0] or canvas[1]<native[1]:
        raise ValueError('Output canvas cannot be smaller than source; no resizing allowed')
    if record.get('evaluation_roi')!=[0,0,native[0],native[1]]:
        raise ValueError('Only recorded native top-left ROI [0,0,width,height] is supported')
    padding=record.get('padding')
    expected={'left':0,'top':0,'right':canvas[0]-native[0],'bottom':canvas[1]-native[1]}
    if not isinstance(padding,dict) or any(padding.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Inconsistent padding metadata; expected {expected}')
    if padding.get('mode') not in {'edge','none'} or (padding['mode']=='none' and any(expected.values())):
        raise ValueError('Only explicit edge padding or no padding is supported')
    return native,canvas


def _alpha_stats(alpha):
    return {'minimum':int(alpha.min()),'maximum':int(alpha.max()),
            'fraction_below_128':float(np.mean(alpha<128)),
            'fraction_below_250':float(np.mean(alpha<250))}


def _alpha_diagnostics(pixels,native):
    """Record native ROI and full-canvas alpha statistics; never gate or change pixels."""
    alpha=pixels[:,:,3] if pixels.shape[2]==4 else np.full(pixels.shape[:2],255,np.uint8)
    canvas_stats=_alpha_stats(alpha)
    roi_stats=_alpha_stats(alpha[:native[1],:native[0]])
    return {'scope':'native_evaluation_roi',**roi_stats},canvas_stats


def prepare_rgb_record(raw_output, rgb_output, metadata):
    """Export exact RGB channels separately; retain raw bytes and diagnose alpha.

    Required metadata: id, source_size, canvas_size, evaluation_roi, padding,
    source_sha256, instruction_sha256. Extra seed/model/steps metadata is retained.
    """
    record=dict(metadata)
    if not isinstance(record.get('id'),str) or not record['id']:raise ValueError('Record id required')
    for key in ['source_sha256','instruction_sha256']:
        if not isinstance(record.get(key),str) or len(record[key])!=64:raise ValueError(f'{key} required')
    native,canvas=validate_geometry(record)
    raw_path,rgb_path=Path(raw_output).resolve(),Path(rgb_output).resolve()
    before=sha256_file(raw_path)
    record.update(adapter_version=ADAPTER_VERSION,raw_output=str(raw_path),raw_sha256=before)
    with Image.open(raw_path) as image:
        image.load();record.update(raw_mode=image.mode,raw_size=list(image.size))
        if image.size!=canvas:raise ValueError(f'Actual output {image.size} != requested canvas {canvas}')
        if image.mode not in {'RGB','RGBA'}:raise ValueError(f'Unsupported raw mode: {image.mode}')
        pixels=np.asarray(image)
        record['alpha'],record['alpha_canvas']=_alpha_diagnostics(pixels,native)
        record['alpha_policy']=ALPHA_POLICY
        if raw_path==rgb_path:
            if image.mode!='RGB':raise ValueError('RGBA export needs separate path; raw output is immutable')
        else:
            rgb_path.parent.mkdir(parents=True,exist_ok=True)
            temp=rgb_path.with_name(rgb_path.name+'.tmp')
            Image.fromarray(pixels[:,:,:3].copy()).save(temp,format='PNG');temp.replace(rgb_path)
        with Image.open(rgb_path) as exported:
            if exported.mode!='RGB' or exported.size!=canvas or not np.array_equal(np.asarray(exported),pixels[:,:,:3]):
                raise ValueError('RGB export did not preserve raw RGB channels')
    if sha256_file(raw_path)!=before:raise RuntimeError('Raw output changed during export')
    record.update(status='complete',rgb_output=str(rgb_path),rgb_sha256=sha256_file(rgb_path),
        export_policy='RGB channels exact; no compositing, crop or resampling; ROI applied only by evaluator')
    return record


def load_generation_records(manifest_path):
    text=Path(manifest_path).read_text()
    try:value=json.loads(text)
    except json.JSONDecodeError:value=[json.loads(l) for l in text.splitlines() if l.strip()]
    if isinstance(value,dict):
        if value.get('schema_version',1)!=1:raise ValueError('Unsupported generation manifest schema_version')
        value=value.get('records')
    if not isinstance(value,list):raise ValueError('Manifest needs records list or JSONL records')
    records={}
    for r in value:
        if not isinstance(r,dict) or not isinstance(r.get('id'),str):raise ValueError('Each record needs string id')
        if r['id'] in records:raise ValueError(f'Duplicate generation record id: {r["id"]}')
        records[r['id']]=r
    return records


def _record_path(value,manifest_path,name):
    if not isinstance(value,str) or not value:raise ValueError(f'Missing {name}')
    path=Path(value)
    return path if path.is_absolute() else Path(manifest_path).resolve().parent/path


def load_prediction(task,dataset_root,*,predictions_dir=None,record=None,manifest_path=None):
    """Return (native RGB numpy array,provenance), without rewriting any file."""
    source_path=Path(dataset_root)/task['image']
    with Image.open(source_path) as source:
        if source.mode!='RGB':raise ValueError(f'Dataset source not RGB: {source.mode}')
        source_size=source.size
    source_sha=sha256_file(source_path)
    instruction_sha=instruction_sha256(task)
    if record is None:
        if predictions_dir is None:raise ValueError('Prediction missing from generation manifest')
        path=Path(predictions_dir)/task.get('output_filename',task['id']+'.png')
        with Image.open(path) as image:
            if image.mode not in {'RGB','RGBA'}:
                raise ValueError(f'Expected RGB or RGBA; got {image.mode}. Use an explicit adapter record/export.')
            if image.size!=source_size:raise ValueError(f'Dimension mismatch: prediction={image.size}, expected={source_size}; use verified padding manifest, no resizing applied')
            rgba=image.mode=='RGBA';pixels=np.asarray(image)
            result=pixels[:,:,:3].copy()
            if rgba:alpha_roi,alpha_canvas=_alpha_diagnostics(pixels,source_size)
        output_sha=sha256_file(path)
        provenance={'mode':'native_rgba_rgb_channels' if rgba else 'native_rgb',
                       'raw_output':str(path.resolve()),'raw_sha256':output_sha,
                       'rgb_output':str(path.resolve()),'rgb_sha256':output_sha,
                       'source_sha256':source_sha,'instruction_sha256':instruction_sha,
                       'source_size':list(source_size),'canvas_size':list(source_size),
                       'evaluation_roi':[0,0,*source_size],
                       'padding':{'left':0,'top':0,'right':0,'bottom':0,'mode':'none'},
                       'adapter_version':None,'validation_adapter_version':ADAPTER_VERSION,
                       'generation_protocol_sha256':None}
        if rgba:
            provenance.update(raw_mode='RGBA',alpha_policy=ALPHA_POLICY,
                              alpha_evaluation_roi=alpha_roi,alpha_canvas=alpha_canvas)
        return result,provenance
    if manifest_path is None:raise ValueError('manifest_path required with record')
    if record.get('id')!=task['id']:raise ValueError('Generation id does not match task')
    if record.get('status')!='complete':raise ValueError(f'Generation status={record.get("status")}: {record.get("error","not complete")}')
    if record.get('adapter_version') not in READABLE_ADAPTER_VERSIONS:
        raise ValueError('Unsupported generation adapter_version')
    native,canvas=validate_geometry(record,source_size)
    if record.get('source_sha256')!=source_sha:raise ValueError('Source hash does not match generation provenance')
    if record.get('instruction_sha256')!=instruction_sha:raise ValueError('Instruction changed since generation; stale output rejected')
    raw_path=_record_path(record.get('raw_output'),manifest_path,'raw_output')
    path=_record_path(record.get('rgb_output'),manifest_path,'rgb_output')
    if record.get('raw_sha256')!=sha256_file(raw_path):raise ValueError('Raw output hash mismatch')
    if record.get('rgb_sha256')!=sha256_file(path):raise ValueError('RGB export hash mismatch')
    model_input={}
    if 'model_input' in record or 'model_input_sha256' in record:
        model_input_path=_record_path(record.get('model_input'),manifest_path,'model_input')
        if record.get('model_input_sha256')!=sha256_file(model_input_path):
            raise ValueError('Model input hash mismatch')
        model_input={'model_input':str(model_input_path.resolve()),
                     'model_input_sha256':record['model_input_sha256']}
    with Image.open(raw_path) as raw,Image.open(path) as image:
        if raw.size!=canvas or image.size!=canvas:raise ValueError('Recorded canvas differs from raw/RGB output size')
        if image.mode!='RGB' or raw.mode not in {'RGB','RGBA'}:raise ValueError('Invalid raw or exported image mode')
        pixels=np.asarray(image);raw_pixels=np.asarray(raw)
        if not np.array_equal(pixels,raw_pixels[:,:,:3]):raise ValueError('RGB export differs from raw RGB channels')
        alpha_roi,alpha_canvas=_alpha_diagnostics(raw_pixels,native)
        result=pixels[:native[1],:native[0]].copy()
    provenance={'mode':'recorded_padding_roi','manifest':str(Path(manifest_path).resolve()),
        'raw_output':str(raw_path.resolve()),'raw_sha256':record['raw_sha256'],
        'rgb_output':str(path.resolve()),'rgb_sha256':record['rgb_sha256'],
        'source_sha256':source_sha,'instruction_sha256':instruction_sha,
        'source_size':list(native),'canvas_size':list(canvas),'evaluation_roi':record['evaluation_roi'],
        'padding':record['padding'],'adapter_version':record.get('adapter_version'),
        'validation_adapter_version':ADAPTER_VERSION,'alpha_policy':ALPHA_POLICY,
        'alpha_evaluation_roi':alpha_roi,'alpha_canvas':alpha_canvas,
        'generation_protocol_sha256':record.get('protocol_sha256'),**model_input}
    return result,provenance
