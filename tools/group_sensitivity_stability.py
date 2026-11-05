import csv, hashlib, json, random, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.UNetFamily import U_Net
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2
from tools.group_trainable_bits import replace_conv_layers_with_groups

SEED0=2026; GROUP_SIZE=8; PATCH_SIZE=64; BATCH_SIZE=64
TRAIN=ROOT/'lists/DRIVE/train_images_seed2026.txt'
CKPT=ROOT/'experiments/DRIVE_Unet_GroupBits_stage1_full_seed2026/best_checkpoint.pt'
OUT=ROOT/'experiments/DRIVE_Unet_GroupBits_structural_sensitivity_stability'
CONFIGS=[(320,2026),(320,2027),(1024,2026),(1024,2027)]

def sha(path):
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def losses(logits,target,fov):
    valid=fov>0; ce=F.cross_entropy(logits,target,reduction='none')
    all_l=ce[valid].mean(); den=F.avg_pool2d((target==1).float().unsqueeze(1),3,1,1).squeeze(1)
    thin=(target==1)&valid&(den<=(4.0/9.0)); thin_l=ce[thin].mean() if thin.any() else all_l*0
    return all_l,thin_l

def rank(vals):
    order=np.argsort(vals,kind='mergesort'); out=np.empty(len(vals),float); out[order]=np.arange(len(vals),dtype=float)
    return out

def spearman(a,b):
    x=rank(np.asarray(a)); y=rank(np.asarray(b));
    return float(np.corrcoef(x,y)[0,1])

def main():
    if OUT.exists(): raise SystemExit(f'OUTPUT_EXISTS: {OUT}')
    for p in (TRAIN,CKPT):
        if not p.exists(): raise FileNotFoundError(p)
    txt=TRAIN.read_text()
    if 'val_images' in txt.lower() or 'test.txt' in txt.lower() or '/test/' in txt.lower(): raise RuntimeError('forbidden manifest reference')
    OUT.mkdir()
    (OUT/'calibration_manifest_used.txt').write_text(txt)
    (OUT/'calibration_manifest_used.txt.sha256').write_text(sha(TRAIN)+'  calibration_manifest_used.txt\n')
    random.seed(SEED0); np.random.seed(SEED0); torch.manual_seed(SEED0)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    imgs,masks,fovs=data_preprocess(str(TRAIN))
    ck=torch.load(CKPT,map_location='cpu'); model=U_Net(1,2); wrappers=replace_conv_layers_with_groups(model,GROUP_SIZE)
    model.load_state_dict(ck.get('net',ck.get('state_dict',ck)),strict=True); model.to(device).eval()
    if sum(w.num_groups for w in wrappers.values())!=857: raise RuntimeError('expected 857 groups')
    all_rows=[]; by_name={}
    for n_patches,seed in CONFIGS:
        args=type('Args',(),{'train_patch_height':PATCH_SIZE,'train_patch_width':PATCH_SIZE,'N_patches':n_patches,'inside_FOV':'center'})()
        idx=create_patch_idx(fovs,args,seed=seed); ds=TrainDatasetV2(imgs,masks,fovs,idx,mode='val',args=args)
        batches=[(x.to(device),y.to(device),f.to(device)) for x,y,f in DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)]
        with torch.no_grad():
            ba=[]; bt=[]
            for x,y,f in batches:
                a,t=losses(model(x),y,f); ba.append(float(a)); bt.append(float(t))
        base_a=float(np.mean(ba)); base_t=float(np.mean(bt)); rows=[]
        with torch.no_grad():
            for layer,w in wrappers.items():
                for gid,(start,end) in enumerate(w.group_slices):
                    original=float(w.bit_logit[gid].item()); p=float(torch.sigmoid(w.bit_logit[gid]).item()); w.bit_logit[gid].fill_(-20.)
                    aa=[]; tt=[]
                    for x,y,f in batches:
                        a,t=losses(model(x),y,f); aa.append(float(a)); tt.append(float(t))
                    w.bit_logit[gid].fill_(original)
                    rows.append({'layer_name':layer,'group_id':gid,'channel_start':start,'channel_end':end-1,'original_bit':4+4*p,'forced_int4':True,'delta_loss_all':float(np.mean(aa)-base_a),'delta_loss_thin':float(np.mean(tt)-base_t)})
        scale=max([max(0.,r['delta_loss_thin']) for r in rows] or [1.])
        for r in rows: r['normalized_sensitivity']=max(0.,r['delta_loss_thin'])/scale
        rows.sort(key=lambda r:r['normalized_sensitivity'],reverse=True)
        tag=f'patches{n_patches}_seed{seed}'
        with (OUT/f'{tag}.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        by_name[tag]=rows
        for r in rows: r.update({'setting':tag,'patches':n_patches,'seed':seed})
        all_rows.extend(rows)
        print('DONE',tag,flush=True)
    with (OUT/'sensitivity_stability_summary.csv').open('w',newline='') as f:
        fields=['setting','patches','seed','top10_overlap_to_320_2026','top20_overlap_to_320_2026','top50_overlap_to_320_2026','spearman_to_320_2026','int8_count','int8_mean_sensitivity','int4_mean_sensitivity','train_manifest_sha256']
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); ref=by_name['patches320_seed2026']; ref_names={(r['layer_name'],r['group_id']) for r in ref}
        refv=[r['normalized_sensitivity'] for r in ref]
        for tag,rows in by_name.items():
            vals=[r['normalized_sensitivity'] for r in rows];
            out={'setting':tag,'patches':tag.split('_')[0].replace('patches',''),'seed':tag.split('seed')[1]}
            for k in (10,20,50): out[f'top{k}_overlap_to_320_2026']=len(ref_names.intersection({(r['layer_name'],r['group_id']) for r in rows[:k]}))/k
            out['spearman_to_320_2026']=spearman(refv,vals); out['int8_count']=sum(r['original_bit']>=6 for r in rows)
            out['int8_mean_sensitivity']=float(np.mean([r['normalized_sensitivity'] for r in rows if r['original_bit']>=6])); out['int4_mean_sensitivity']=float(np.mean([r['normalized_sensitivity'] for r in rows if r['original_bit']<6])); out['train_manifest_sha256']=sha(TRAIN); wr.writerow(out)
    (OUT/'all_group_sensitivity.csv').write_text('')
    with (OUT/'all_group_sensitivity.csv').open('w',newline='') as f:
        fields=list(all_rows[0]); wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(all_rows)
    top_layers=[]
    for tag,rows in by_name.items(): top_layers.append((tag,[r['layer_name'] for r in rows[:20]]))
    meta={'checkpoint':str(CKPT.relative_to(ROOT)),'checkpoint_epoch':int(ck.get('epoch',-1)),'group_size':GROUP_SIZE,'configs':[{'patches':p,'seed':s} for p,s in CONFIGS],'train_manifest':str(TRAIN.relative_to(ROOT)),'train_manifest_sha256':sha(TRAIN),'validation_or_test_used':False,'checkpoint_modified':False,'bit_logits_modified':False,'normalization':'max positive DeltaL_thin per setting','thin_proxy':'3x3 local vessel ratio <= 4/9'}
    (OUT/'README.md').write_text('Train-only sensitivity stability validation. Four independent patch/seed settings; no training, checkpoint modification, bit-logit update, validation/test data, or test inference.\n\n'+json.dumps(meta,indent=2)+'\n\nTop-20 layers by setting:\n'+json.dumps(top_layers,indent=2))
    plt.figure(figsize=(8,4));
    for tag,rows in by_name.items(): plt.plot([r['normalized_sensitivity'] for r in rows[:50]],label=tag)
    plt.xlabel('rank'); plt.ylabel('normalized sensitivity'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT/'sensitivity_stability.png',dpi=160); plt.close()
    print(json.dumps({'output':str(OUT),'settings':list(by_name),'top10':{k:[(r['layer_name'],r['group_id']) for r in v[:10]] for k,v in by_name.items()}},indent=2),flush=True)

if __name__=='__main__': main()
