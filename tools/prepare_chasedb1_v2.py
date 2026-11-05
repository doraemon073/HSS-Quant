#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def ids_for(split_dir, split):
    path=split_dir/(split+'_ids_seed2026.txt')
    if not path.exists(): raise FileNotFoundError(path)
    ids=[x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate IDs: '+str(path))
    return ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--split-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args()
    root=a.dataset_root.resolve(); split_dir=a.split_dir.resolve(); out=a.output_dir.resolve()
    image_dir=root/'CHASEDB1'
    if not image_dir.is_dir(): raise FileNotFoundError(image_dir)
    if out.exists(): raise SystemExit('OUTPUT_EXISTS: '+str(out))
    out.mkdir(parents=True); fov_dir=out/'fov_masks'; fov_dir.mkdir()
    audit={'dataset_root':str(root),'label':'1stHO','fov_rule':'np.any(rgb > 0, axis=2)','splits':{}}
    for split in ('train','val','test'):
        ids=ids_for(split_dir,split); rows=[]
        for image_id in ids:
            image_path=image_dir/(image_id+'.jpg'); mask_path=image_dir/(image_id+'_1stHO.png')
            if not image_path.exists() or not mask_path.exists(): raise FileNotFoundError(str(image_path)+' '+str(mask_path))
            with Image.open(image_path) as im: rgb=np.asarray(im.convert('RGB'))
            with Image.open(mask_path) as im: mask=np.asarray(im.convert('L'))
            if rgb.shape[:2]!=(960,999) or mask.shape!=(960,999): raise ValueError(f'shape {image_id}: {rgb.shape} {mask.shape}')
            fov=(np.any(rgb>0,axis=2).astype(np.uint8)*255)
            fov_path=fov_dir/(image_id+'_fov.png'); Image.fromarray(fov,mode='L').save(fov_path)
            if int(fov.min())!=0 or int(fov.max())!=255: raise ValueError('bad FOV range '+image_id)
            rows.append((str(image_path),str(mask_path),str(fov_path)))
        list_path=out/(split+'.txt'); list_path.write_text(''.join(' '.join(row)+'\\n' for row in rows),encoding='utf-8')
        audit['splits'][split]={'count':len(rows),'ids':ids,'path_list':str(list_path),'path_list_sha256':sha256(list_path)}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# CHASE_DB1 prepared metadata\\n\\nOriginal images and 1stHO annotations are referenced in place and were not modified.\\n\\nFOV masks are generated with np.any(rgb > 0, axis=2) and saved as 0/255 PNGs. Visually audit this rule before official evaluation.\\n\\nPath lists use the fixed subject-level seed-2026 split.\\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
