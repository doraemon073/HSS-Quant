import csv, json, random, hashlib, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from models.UNetFamily import U_Net
from lib.datasetV2 import data_preprocess,create_patch_idx,TrainDatasetV2
from tools.group_trainable_bits import replace_conv_layers_with_groups

TRAIN=ROOT/'lists/DRIVE/train_images_seed2026.txt'; CKPT=ROOT/'experiments/DRIVE_Unet_GroupBits_stage1_full_seed2026/best_checkpoint.pt'; OUT=ROOT/'experiments/DRIVE_Unet_GroupBits_structural_sensitivity_stability'; PATCH=64; BATCH=256; G=8
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def loss(logits,y,f):
 v=f>0; ce=F.cross_entropy(logits,y,reduction='none'); a=ce[v].mean(); d=F.avg_pool2d((y==1).float().unsqueeze(1),3,1,1).squeeze(1); m=(y==1)&v&(d<=(4/9)); t=ce[m].mean() if m.any() else a*0; return a,t
def rank(x):
 o=np.argsort(x,kind='mergesort'); z=np.empty(len(x),float); z[o]=np.arange(len(x)); return z
def spear(a,b): return float(np.corrcoef(rank(np.array(a)),rank(np.array(b)))[0,1])
def readcsv(p): return list(csv.DictReader(p.open()))
def main():
 if not OUT.exists(): raise SystemExit('OUTPUT_MISSING')
 if (OUT/'patches1024_seed2027.csv').exists(): raise SystemExit('FOURTH_RESULT_EXISTS')
 text=TRAIN.read_text();
 if 'val_images' in text.lower() or 'test.txt' in text.lower() or '/test/' in text.lower(): raise RuntimeError('forbidden manifest reference')
 random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 imgs,masks,fovs=data_preprocess(str(TRAIN)); args=type('A',(),{'train_patch_height':PATCH,'train_patch_width':PATCH,'N_patches':1024,'inside_FOV':'center'})(); idx=create_patch_idx(fovs,args,seed=2027); ds=TrainDatasetV2(imgs,masks,fovs,idx,mode='val',args=args); batches=[(x.to(dev),y.to(dev),f.to(dev)) for x,y,f in DataLoader(ds,batch_size=BATCH,shuffle=False,num_workers=0)]
 ck=torch.load(CKPT,map_location='cpu'); model=U_Net(1,2); wraps=replace_conv_layers_with_groups(model,G); model.load_state_dict(ck.get('net',ck.get('state_dict',ck)),strict=True); model.to(dev).eval()
 with torch.no_grad():
  aa=[];tt=[]
  for x,y,f in batches: a,t=loss(model(x),y,f); aa.append(float(a));tt.append(float(t))
 ba=float(np.mean(aa));bt=float(np.mean(tt)); rows=[]
 with torch.no_grad():
  for layer,w in wraps.items():
   for gid,(st,en) in enumerate(w.group_slices):
    old=float(w.bit_logit[gid].item()); p=float(torch.sigmoid(w.bit_logit[gid]).item()); w.bit_logit[gid].fill_(-20.); aa=[];tt=[]
    for x,y,f in batches: a,t=loss(model(x),y,f);aa.append(float(a));tt.append(float(t))
    w.bit_logit[gid].fill_(old); rows.append({'layer_name':layer,'group_id':gid,'channel_start':st,'channel_end':en-1,'original_bit':4+4*p,'forced_int4':True,'delta_loss_all':float(np.mean(aa)-ba),'delta_loss_thin':float(np.mean(tt)-bt)})
 scale=max([max(0.,r['delta_loss_thin']) for r in rows] or [1.]); [r.update({'normalized_sensitivity':max(0.,r['delta_loss_thin'])/scale}) for r in rows]; rows.sort(key=lambda r:r['normalized_sensitivity'],reverse=True)
 with (OUT/'patches1024_seed2027.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 files=['patches320_seed2026.csv','patches320_seed2027.csv','patches1024_seed2026.csv','patches1024_seed2027.csv']; allr={p[:-4]:readcsv(OUT/p) for p in files}; ref=allr['patches320_seed2026']; refset={(r['layer_name'],r['group_id']) for r in ref}; refv=[float(r['normalized_sensitivity']) for r in ref]
 with (OUT/'sensitivity_stability_summary.csv').open('w',newline='') as f:
  fs=['setting','patches','seed','top10_overlap_to_320_2026','top20_overlap_to_320_2026','top50_overlap_to_320_2026','spearman_to_320_2026','int8_count','int8_mean_sensitivity','int4_mean_sensitivity','train_manifest_sha256'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader()
  for tag,rs in allr.items():
   d={'setting':tag,'patches':tag.split('_')[0].replace('patches',''),'seed':tag.split('seed')[1]};
   for k in (10,20,50): d[f'top{k}_overlap_to_320_2026']=len(refset.intersection({(r['layer_name'],r['group_id']) for r in rs[:k]}))/k
   vals=[float(r['normalized_sensitivity']) for r in rs]; d['spearman_to_320_2026']=spear(refv,vals); d['int8_count']=sum(float(r['original_bit'])>=6 for r in rs); d['int8_mean_sensitivity']=float(np.mean([float(r['normalized_sensitivity']) for r in rs if float(r['original_bit'])>=6])); d['int4_mean_sensitivity']=float(np.mean([float(r['normalized_sensitivity']) for r in rs if float(r['original_bit'])<6])); d['train_manifest_sha256']=sha(TRAIN);w.writerow(d)
 meta={'checkpoint':str(CKPT.relative_to(ROOT)),'checkpoint_epoch':int(ck.get('epoch',-1)),'group_size':G,'settings':[{'patches':320,'seed':2026},{'patches':320,'seed':2027},{'patches':1024,'seed':2026},{'patches':1024,'seed':2027}],'train_manifest':str(TRAIN.relative_to(ROOT)),'train_manifest_sha256':sha(TRAIN),'validation_or_test_used':False,'checkpoint_modified':False,'bit_logits_modified':False,'normalization':'max positive DeltaL_thin per setting','thin_proxy':'3x3 local vessel ratio <= 4/9'}
 (OUT/'README.md').write_text('Train-only structural sensitivity stability validation. The fourth setting was resumed after the remote session disconnected; existing CSVs were preserved. No training, checkpoint modification, bit-logit update, validation/test data, or test inference.\n\n'+json.dumps(meta,indent=2))
 plt.figure(figsize=(8,4));
 for tag,rs in allr.items(): plt.plot([float(r['normalized_sensitivity']) for r in rs[:50]],label=tag)
 plt.xlabel('rank');plt.ylabel('normalized sensitivity');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(OUT/'sensitivity_stability.png',dpi=160);plt.close();print('STABILITY_COMPLETE')
if __name__=='__main__':main()
