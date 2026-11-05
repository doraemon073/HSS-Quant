import argparse, csv, hashlib, json, random, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.UNetFamily import U_Net
from lib.common import setpu_seed
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2
from lib.extract_patches import load_file_path_txt
from tools.group_trainable_bits import replace_conv_layers_with_groups
from tools.group_trainable_bits import GroupTrainableBitConv2d, _tensor_fake_quant, _symmetric_weight_fake_quant

def fast_group_forward(self, inputs):
 x4=_tensor_fake_quant(inputs,-8,7); x8=_tensor_fake_quant(inputs,-128,127)
 w4=_symmetric_weight_fake_quant(self.weight,7); w8=_symmetric_weight_fake_quant(self.weight,127)
 y4=F.conv2d(x4,w4,None if self.bias is None else self.bias,self.stride,self.padding,self.dilation,self.groups)
 y8=F.conv2d(x8,w8,None if self.bias is None else self.bias,self.stride,self.padding,self.dilation,self.groups)
 p=torch.cat([self.bit_probability()[i].expand(e-s) for i,(s,e) in enumerate(self.group_slices)])
 return y4*(1-p.view(1,-1,1,1))+y8*p.view(1,-1,1,1)
GroupTrainableBitConv2d.forward=fast_group_forward

ROOT=Path(__file__).resolve().parents[1]
TRAIN=Path('/data/xpp/data/CHASE_DB1-main/CHASE_DB1-main/prepared_seed2026/train.txt'); VAL=Path('/data/xpp/data/CHASE_DB1-main/CHASE_DB1-main/prepared_seed2026/val.txt')
FP32=ROOT/'experiments/CHASEDB1_Unet_FP32_seed2026/best_model.pth'
SENS=ROOT/'experiments/CHASEDB1_GroupBits_structural_sensitivity_seed2026_retry/group_sensitivity.csv'
TARGET=4.269841064; TOL=0.01
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def rows_from(ws):
 out=[]
 for n,w in ws.items():
  e=w.weight.shape[1]*w.weight.shape[2]*w.weight.shape[3]
  for gid,(s,t) in enumerate(w.group_slices): out.append({'key':f'{n}:{gid}','layer_name':n,'group_id':gid,'channel_start':s,'channel_end':t-1,'weight_numel':int((t-s)*e)})
 return out
def stats(rows,a):
 total=sum(r['weight_numel'] for r in rows); bits=sum(r['weight_numel']*a[r['key']] for r in rows)
 return {'int4_groups':sum(a[r['key']]==4 for r in rows),'int8_groups':sum(a[r['key']]==8 for r in rows),'parameter_weighted_mean_bitwidth':bits/total,'ordinary_mean_bitwidth':sum(a[r['key']] for r in rows)/len(rows),'theoretical_conv2d_weight_bytes':bits/8,'theoretical_compression_ratio':total*4/(bits/8)}
def choose(rows,ranked,target):
 total=sum(r['weight_numel'] for r in rows); goal=(target-4)*total/4; w={r['key']:r['weight_numel'] for r in rows}; pos={r['key']:i for i,r in enumerate(ranked)}; chosen=set(); cur=0
 def obj(s,x): return (abs(x-goal),sum(pos[k] for k in s))
 for r in ranked:
  if obj(chosen|{r['key']},cur+w[r['key']])<obj(chosen,cur): chosen.add(r['key']); cur+=w[r['key']]
 while 1:
  base=obj(chosen,cur); best=(base,chosen,cur)
  for r in rows:
   k=r['key']; s=chosen-{k} if k in chosen else chosen|{k}; x=cur-w[k] if k in chosen else cur+w[k]; c=(obj(s,x),s,x)
   if c[0]<best[0]: best=c
  for xk in list(chosen):
   for r in rows:
    if r['key'] in chosen: continue
    s=(chosen-{xk})|{r['key']}; x=cur-w[xk]+w[r['key']]; c=(obj(s,x),s,x)
    if c[0]<best[0]: best=c
  if best[0]>=base: return {r['key']:(8 if r['key'] in chosen else 4) for r in rows}
  _,chosen,cur=best
def loader(manifest,n,batch,seed,shuffle):
 imgs,masks,fovs=data_preprocess(str(manifest)); a=argparse.Namespace(train_patch_height=64,train_patch_width=64,N_patches=n,inside_FOV='center'); idx=create_patch_idx(fovs,a,seed=seed); ds=TrainDatasetV2(imgs,masks,fovs,idx,mode='val',args=a); return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--method',choices=['learned_g8','structural'],required=True); p.add_argument('--output-dir',required=True); a=p.parse_args(); out=(ROOT/a.output_dir).resolve()
 if out.exists(): raise FileExistsError(out)
 for q in (TRAIN,VAL,FP32,SENS):
  if not q.is_file(): raise FileNotFoundError(q)
 tr=load_file_path_txt(str(TRAIN)); va=load_file_path_txt(str(VAL));
 if len(tr[0])!=16 or len(va[0])!=4: raise RuntimeError('expected fixed 16/4 split')
 setpu_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 ck=torch.load(FP32,map_location='cpu',weights_only=True); model=U_Net(1,2); model.load_state_dict(ck['net'],strict=True); ws=replace_conv_layers_with_groups(model,8); model.to(dev)
 for w in ws.values(): w.weight.requires_grad_(False); w.bias is None or w.bias.requires_grad_(False)
 rows=rows_from(ws); assert len(rows)==857; train=loader(TRAIN,5000,64,a.seed,True); val=loader(VAL,1250,64,a.seed+1,False); opt=torch.optim.Adam([w.bit_logit for w in ws.values()],lr=.01); out.mkdir(parents=False); log_path=out/'training_log.csv'; log=[]; best=-1; best_state=None; started=time.time()
 for epoch in range(1,21):
  model.train(); total=cnt=0
  for x,y,f in train:
   x,y,f=x.to(dev),y.to(dev),f.to(dev); opt.zero_grad(set_to_none=True); z=model(x); m=f>0; ce=F.cross_entropy(z.permute(0,2,3,1)[m],y[m]); bit=torch.stack([w.continuous_bits().mean() for w in ws.values()]).mean(); loss=ce+.001*bit
   if a.method=='structural':
    d=F.avg_pool2d((y==1).float().unsqueeze(1),3,1,1).squeeze(1); tm=(y==1)&m&(d<=4/9); ww=torch.where(tm,torch.tensor(4.,device=dev),torch.ones_like(z[:,0])); ww=torch.where((y==1)&m,torch.maximum(ww,torch.tensor(2.,device=dev)),ww); loss=loss+(F.cross_entropy(z,y,reduction='none')*ww*m.float()).sum()/ww[m].sum().clamp_min(1)
   loss.backward(); opt.step(); total+=float(loss)*int(m.sum()); cnt+=int(m.sum())
  model.eval(); ys=[]; ps=[]; vl=vn=0
  with torch.no_grad():
   for x,y,f in val:
    x,y,f=x.to(dev),y.to(dev),f.to(dev); z=model(x); m=f>0; l=F.cross_entropy(z.permute(0,2,3,1)[m],y[m]); vl+=float(l)*int(m.sum()); vn+=int(m.sum()); ys.append(y[m].cpu().numpy()); ps.append(z[:,1][m].sigmoid().cpu().numpy())
  from sklearn.metrics import f1_score,roc_auc_score
  yy=np.concatenate(ys); pp=np.concatenate(ps); f1=float(f1_score(yy,pp>=.5)); auc=float(roc_auc_score(yy,pp)); wb=sum(float(w.parameter_weighted_average_bits().detach())*w.weight.numel() for w in ws.values())/sum(w.weight.numel() for w in ws.values()); row={'epoch':epoch,'train_loss':total/max(cnt,1),'validation_loss':vl/max(vn,1),'validation_f1':f1,'validation_auc':auc,'parameter_weighted_average_bits':wb}; log.append(row)
  with log_path.open('a',newline='') as lf:
   wlog=csv.DictWriter(lf,fieldnames=list(row))
   if lf.tell()==0: wlog.writeheader()
   wlog.writerow(row)
  current={'epoch':epoch,'net':model.state_dict(),'metrics':row,'test_used':False,'method':a.method,'seed':a.seed,'config':vars(a)}
  torch.save(current,out/'latest_checkpoint.pt')
  if f1>best:
   best=f1; best_state=current; torch.save(best_state,out/'best_checkpoint.pt')
 torch.save(best_state,out/'best_checkpoint.pt')
 torch.save(best_state,out/'best_checkpoint.pt'); torch.save(current,out/'latest_checkpoint.pt'); model.load_state_dict(best_state['net']); byname=dict(ws); learned={r['key']:(8 if float(torch.sigmoid(byname[r['layer_name']].bit_logit[r['group_id']]))>=.5 else 4) for r in rows}
 if a.method=='structural': sens={f"{r['layer_name']}:{r['group_id']}":r for r in csv.DictReader(SENS.open())}; ranked=sorted(rows,key=lambda r:float(sens[r['key']]['normalized_sensitivity']),reverse=True)
 else: ranked=sorted(rows,key=lambda r:float(torch.sigmoid(byname[r['layer_name']].bit_logit[r['group_id']])),reverse=True)
 assignment=choose(rows,ranked,TARGET); st=stats(rows,assignment); assert abs(st['parameter_weighted_mean_bitwidth']-TARGET)<=TOL
 payload={'method':a.method,'seed':a.seed,'group_size':8,'total_groups':857,**st,'target_weighted_mean_bitwidth':TARGET,'absolute_budget_error':abs(st['parameter_weighted_mean_bitwidth']-TARGET),'assignment':assignment,'selected_group_names':[k for k,v in assignment.items() if v==8],'budget_match_passed':True}; (out/'assignment.json').write_text(json.dumps(payload,indent=2));
 with (out/'group_assignment.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())+['bit']); w.writeheader(); [w.writerow({**r,'bit':assignment[r['key']]}) for r in rows]
 with (out/'training_log.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
 cfg={'method':a.method,'seed':a.seed,'epochs':20,'training_patches':5000,'batch_size':64,'calibration_patches':320,'learning_rate_bits':.01,'lambda_bit':.001,'group_size':8,'group_count':857,'train_manifest':str(TRAIN),'validation_manifest':str(VAL),'train_manifest_sha256':sha(TRAIN),'validation_manifest_sha256':sha(VAL),'checkpoint':str(FP32),'checkpoint_sha256':sha(FP32),'test_used':False,'test_manifest_not_read':True,'training_seconds':time.time()-started,'budget':st}; (out/'quant_config.json').write_text(json.dumps(cfg,indent=2)); (out/'audit.json').write_text(json.dumps({'test_used':False,'training_used':True,'calibration_used':True,'group_count':857,'budget_match_passed':True,'test_manifest_read':False},indent=2)); (out/'README.md').write_text('G=8 group stability training; frozen FP32 weights; test manifest not read.\n'); print(json.dumps({'output':str(out),'method':a.method,'seed':a.seed,'budget':st},indent=2))
if __name__=='__main__': main()
