"""Create clearly labeled comparison previews; raw model outputs are untouched."""
from settings import DATASET_ROOT, MODEL_ROOT
import argparse
import html
import json
from pathlib import Path
import shutil

from PIL import Image,ImageDraw,ImageFont
from output_adapter import load_generation_records,load_prediction
from output_adapter import sha256_file
from evaluation_binding import BINDING_VERSION,binding_version,case_binding,dataset_identity,preview_sidecar,source_files,write_json

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dataset-root',type=Path,default=DATASET_ROOT)
    p.add_argument('--page-size',type=int,default=20,help='Maximum cases per preview PNG; never creates a giant full-dataset sheet')
    a=p.parse_args(argv)
    if a.page_size<1 or a.page_size>50:p.error('--page-size must be between 1 and 50')
    records=load_generation_records(a.generation_manifest)
    dataset=a.dataset_root.resolve()
    tasks=[json.loads(l) for l in (dataset/'tasks.jsonl').read_text().splitlines() if l.strip()]
    tasks=[t for t in tasks if t['id'] in records]
    if not tasks:raise ValueError('No generation records for preview')
    if {t['id'] for t in tasks}!=set(records):raise ValueError('Unknown generation record IDs in preview')
    bindings=[];errors=[];pages=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    widths=[210,360,360,360];rowh=235
    a.output.parent.mkdir(parents=True,exist_ok=True)
    page_directory=a.output.parent/(a.output.stem+'_pages')
    if len(tasks)>a.page_size:page_directory.mkdir(parents=True,exist_ok=True)
    for index,t in enumerate(tasks):
        local_index=index%a.page_size
        if local_index==0:
            count=min(a.page_size,len(tasks)-index)
            sheet=Image.new('RGB',(sum(widths),rowh*count),'#ededed');draw=ImageDraw.Draw(sheet)
        folder=dataset/'cases'/t['id'];ann=json.loads((folder/'annotation.json').read_text())
        source=Image.open(folder/'source.png').convert('RGB');target=Image.open(folder/'target.png').convert('RGB')
        boxes=[ann[k]['box'] for k in ['source_geometry','target_geometry']]
        crop=(max(0,int(min(b['x'] for b in boxes)-60)),max(0,int(min(b['y'] for b in boxes)-60)),
            min(source.width,int(max(b['x']+b['width'] for b in boxes)+60)),min(source.height,int(max(b['y']+b['height'] for b in boxes)+60)))
        error=None
        try:
            prediction,provenance=load_prediction(t,dataset,record=records[t['id']],manifest_path=a.generation_manifest)
            bindings.append(case_binding(t,dataset,provenance,preview=True))
            result=Image.fromarray(prediction).crop(crop)
        except Exception as exc:
            result=Image.new('RGB',(widths[-1]-12,rowh-45),'#ffe5e5');error=str(exc)
            errors.append({'id':t['id'],'error':error})
            ImageDraw.Draw(result).text((8,8),records[t['id']].get('status','error'),font=font,fill='black')
        images=[source,source.crop(crop),target.crop(crop),result]
        labels=[t['id']+' / full source','source crop','reference crop','model crop' if not error else 'output rejected/error']
        x=0
        for width,im,label in zip(widths,images,labels):
            im.thumbnail((width-12,rowh-40));sheet.paste(im,(x+(width-im.width)//2,local_index*rowh+28+(rowh-40-im.height)//2))
            draw.text((x+6,local_index*rowh+6),label,font=font,fill='black');x+=width
        if local_index==a.page_size-1 or index==len(tasks)-1:
            page_path=(page_directory/f'page_{len(pages)+1:03}.png') if len(tasks)>a.page_size else a.output
            sheet.save(page_path)
            start=index-local_index
            pages.append({'path':page_path.relative_to(a.output.parent).as_posix(),
                          'sha256':sha256_file(page_path),'case_ids':[x['id'] for x in tasks[start:index+1]]})
    if len(pages)>1:shutil.copy2(a.output.parent/pages[0]['path'],a.output)
    index_path=a.output.with_suffix('.html')
    sections=[]
    for number,page in enumerate(pages,1):
        url=html.escape(page['path'],quote=True)
        ids=', '.join(page['case_ids'])
        sections.append(f'<section><h2>Page {number}</h2><p>{html.escape(ids)}</p>'
                        f'<a href="{url}">Open full PNG</a><img loading="lazy" src="{url}" alt="Page {number}"></section>')
    index_path.write_text('<!doctype html><meta charset="utf-8"><title>Editing comparisons</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#eee}img{display:block;max-width:100%;height:auto}'
        'section{margin:28px 0}p{overflow-wrap:anywhere}</style><h1>Editing comparisons</h1>'
        f'<p>{len(tasks)} cases; {len(pages)} pages. The model output files are preserved separately.</p>'
        +''.join(sections)+'\n')
    write_json(preview_sidecar(a.output),{'version':binding_version(dataset),'complete':not errors,
        'preview_format':'paged-comparisons-v1','page_size':a.page_size,'pages':pages,
        'html_index':{'path':index_path.relative_to(a.output.parent).as_posix(),'sha256':sha256_file(index_path)},
        'dataset':dataset_identity(dataset),'sources':source_files(('preview_run.py','output_adapter.py','evaluation_binding.py')),
        'output_sha256':sha256_file(a.output),'results':bindings,'errors':errors})
    print(a.output)
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
