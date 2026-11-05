import argparse, csv, hashlib, json, os, random, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.UNetFamily import U_Net
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2
from tools.group_trainable_bits import replace_conv_layers_with_groups, group_statistics

SEED = 2026
GROUP_SIZE = 8
N_PATCHES = 32
BATCH_SIZE = 32
PATCH_SIZE = 64
TRAIN_MANIFEST = ROOT / 'lists/DRIVE/train_images_seed2026.txt'
CKPT = ROOT / 'experiments/DRIVE_Unet_GroupBits_stage1_full_seed2026/best_checkpoint.pt'
MAPPING = ROOT / 'experiments/DRIVE_Unet_GroupBits_stage1_full_seed2026/group_mapping.json'
OUT = ROOT / 'experiments/DRIVE_Unet_GroupBits_structural_sensitivity_seed2026'

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def loss_parts(logits, target, fov):
    valid=fov>0
    ce=F.cross_entropy(logits, target, reduction='none')
    all_loss=ce[valid].mean()
    vessel=(target==1).float().unsqueeze(1)
    density=F.avg_pool2d(vessel,3,1,1).squeeze(1)
    thin=(target==1)&valid&(density <= (4.0/9.0))
    thin_loss=ce[thin].mean() if thin.any() else ce[valid].mean()*0.0
    return all_loss, thin_loss

def main():
    if OUT.exists(): raise SystemExit(f'OUTPUT_EXISTS: {OUT}')
    for p in (TRAIN_MANIFEST, CKPT, MAPPING):
        if not p.exists(): raise FileNotFoundError(p)
    text=TRAIN_MANIFEST.read_text()
    if 'test.txt' in text.lower() or '/test/' in text.lower():
        raise RuntimeError('forbidden test reference in train manifest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_imgs, train_masks, train_fovs=data_preprocess(str(TRAIN_MANIFEST))
    args=argparse.Namespace(train_patch_height=PATCH_SIZE, train_patch_width=PATCH_SIZE,
                            N_patches=N_PATCHES, inside_FOV='center')
    idx=create_patch_idx(train_fovs,args,seed=SEED)
    ds=TrainDatasetV2(train_imgs,train_masks,train_fovs,idx,mode='val',args=args)
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    batches=[]
    for x,y,f in loader: batches.append((x.to(device),y.to(device),f.to(device)))
    model=U_Net(1,2)
    ck=torch.load(CKPT,map_location='cpu')
    wrappers=replace_conv_layers_with_groups(model, GROUP_SIZE)
    state=ck.get('net',ck.get('state_dict',ck))
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    if sum(w.num_groups for w in wrappers.values()) != 857: raise RuntimeError('group count mismatch')
    with torch.no_grad():
        base_all=[]; base_thin=[]
        for x,y,f in batches:
            a,t=loss_parts(model(x),y,f); base_all.append(float(a)); base_thin.append(float(t))
    base_all=float(np.mfan(base_all)); base_thin=float(np.mean(base_thin))
    rows=[]
    with torch.no_grad():
        for layer,w in wrappers.items():
            for gid,(start,end) in enumerate(w.group_slices):
                original=float(w.bit_logit[gid].item()); p=float(torch.sigmoid(w.bit_logit[gid]).item())
                w.bit_logit[gid].fill_(-20.0)
                aa=[]; tt=[]
                for x,y,f in batches:
                    a,t=loss_parts(model(x),y,f); aa.append(float(a)); tt.append(float(t))
                w.bit_logit[gid].fill_(original)
                rows.append({'layer_name':layer,'group_id':gid,'channel_start':start,'channel_end':end-1,
                             'original_bit':4.0+4.0*p,'forced_int4':True,
                             'delta_loss_all':float(np.mean(aa)-base_all),
                             'delta_loss_thin':float(np.mean(tt)-base_thin)})
    scale=max([max(0.0,r['delta_loss_thin']) for r in rows] or [1.0])
    for r in rows: r['normalized_sensitivity']=max(0.0,r['delta_loss_thin'])/scale
    rows.sort(key=lambda r:r['normalized_sensitivity'],reverse=True)
    OUT.mkdir(parents=False)
    with (OUT/'group_sensitivity.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    meta={'checkpoint':str(CKPT.relative_to(ROOT)),'checkpoint_epoch':int(ck.get('epoch',-1)),
          'seed':SEED,'group_size':GROUP_SIZE,'calibration_patches':N_PATCHES,
          'train_manifest':str(TRAIN_MANIFEST.relative_to(ROOT)),'train_manifest_sha256':sha256(TRAIN_MANIFEST),
          'validation_or_test_used':False,'baseline_loss_all':base_all,'baseline_loss_thin':base_thin,
          'group_count':len(rows)}
    (OUT/'group_sensitivity.json').write_text(json.dumps({'metadata':meta,'groups':rows},indent=2))
    plt.figure(figsize=(10,4)); plt.plot([r['normalized_sensitivity'] for r in rows]); plt.xlabel('group rank'); plt.ylabel('normalized thin sensitivity'); plt.tight_layout(); plt.savefig(OUT/'sensitivity_summary.png',dpi=160); plt.close()
    (OUT/'calibration_manifest_used.txt').write_text(TRAIN_MANIFEST.read_text())
    (OUT/'calibration_manifest_used.txt.sha256').write_text(sha256(TRAIN_MANIFEST)+'  calibration_manifest_used.txt\n')
    (OUT/'README.md').write_text('Train-only structural sensitivity analysis. G=8 channel groups; 32 train patches; thin proxy is 3x3 local vessel ratio <= 4/9. No validation/test data, retraining, checkpoint modification, skeleton, branch, clDice, or hardware model was used.\n\n'+json.dumps(meta,indent=2))
    print(json.dumps({'high_sensitivity_groups':sum(r['normalized_sensitivity']>=0.5 for r in rows),'max_group':rows[0],'min_group':rows[-1],'output_files':sorted(p.name for p in OUT.iterdir())},indent=2))

if __name__=='__main__': main()
