import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.UNetFamily import U_Net
from lib.extract_patches import load_data,my_PreProc,paint_border_overlap,extract_ordered_overlap,recompone_overlap
from tools.evaluate_group_budget_official_test_v2 import prepare_eval_arrays
ROOT=Path(__file__).resolve().parents[1]
FP32=ROOT/'experiments/DRIVE_Unet_FP32_full_retry/best_model.pth'
INT4=ROOT/'experiments/DRIVE_Unet_INT4_PTQ_baseline/int4_weight_state_dict.pt'
TEST=ROOT/'prepare_dataset/data_path_list/DRIVE/test.txt'

def state(o):
    if isinstance(o,dict):
        for k in ('net','state_dict','model'):
            if isinstance(o.get(k),dict): return o[k]
    return o
def unpack(rec):
    raw=rec['weight_int4_packed'].cpu().numpy().reshape(-1)
    if rec['weight_value_count']>2*raw.size: raise ValueError('packed count exceeds capacity')
    lo=raw&15; hi=(raw>>4)&15
    vals=np.empty(raw.size*2,dtype=np.int8); vals[0::2]=np.where(lo>=8,lo-16,lo); vals[1::2]=np.where(hi>=8,hi-16,hi)
    vals=vals[:rec['weight_value_count']]
    arr=vals.reshape(tuple(rec['weight_shape']))
    return arr
class QConv(nn.Module):
    def __init__(self,c,r):
        super().__init__(); self.conv=c; self.register_buffer('qi',torch.from_numpy(unpack(r)).to(torch.int8)); self.register_buffer('sc',r['weight_scales'].float()); self.register_buffer('zp',r['weight_zero_points'].int()); self.a=float(r['activation_scale']); self.az=int(r['activation_zero_point']); self.r=r
    def forward(self,x):
        x=torch.fake_quantize_per_tensor_affine(x,self.a,self.az,-8,7)
        w=(self.qi.float()-self.zp.float().view(-1,1,1,1))*self.sc.view(-1,1,1,1); p=self.r['conv_parameters']
        return F.conv2d(x,w,self.conv.bias,tuple(p['stride']),tuple(p['padding']),tuple(p['dilation']),int(p['groups']))
def replace(m,L,p=''):
    for n,c in list(m.named_children()):
        full=f'{p}.{n}' if p else n
        if isinstance(c,nn.Conv2d):
            if full not in L: raise KeyError(full)
            r=L[full]; req={'weight_int4_packed','weight_value_count','weight_shape','weight_scales','weight_zero_points','activation_scale','activation_zero_point','conv_parameters'}
            if not req.issubset(r): raise ValueError(f'missing fields {full}')
            a=unpack(r)
            if tuple(a.shape)!=tuple(r['weight_shape']): raise ValueError(f'shape mismatch {full}')
            setattr(m,n,QConv(c,r))
        else: replace(c,L,full)
def run(m,patches,ph,pw,imgshape,mk,fv,dev):
    z=[]
    with torch.no_grad():
        for i in range(0,len(patches),16): z.append(torch.softmax(m(torch.from_numpy(patches[i:i+16]).float().to(dev)),1)[:,1].cpu().numpy())
    pred=recompone_overlap(np.expand_dims(np.concatenate(z),1),ph,pw,16,16)[0]
    return prepare_eval_arrays(pred,mk,fv,imgshape,(0,imgshape[0],0,imgshape[1]))
def stats(a): return [float(a.min()),float(a.max()),float(a.mean()),float(a.std())]
imgs,masks,fovs=load_data(str(TEST)); padded=paint_border_overlap(my_PreProc(imgs[:1]),96,96,16,16); ph,pw=padded.shape[2:]; patches=extract_ordered_overlap(padded,96,96,16,16); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fo=torch.load(FP32,map_location='cpu',weights_only=False); qo=torch.load(INT4,map_location='cpu',weights_only=False); print('ARTIFACT_FIELDS',list(qo), 'LAYERS',len(qo['layers']))
fp=U_Net(1,2); fp.load_state_dict(state(fo),strict=True); q=U_Net(1,2); q.load_state_dict(state(fo),strict=True); replace(q,qo['layers']); fp.to(dev).eval(); q.to(dev).eval()
fs,tr,fm=run(fp,patches,ph,pw,imgs.shape[2:],masks[0,0],fovs[0,0],dev); qs,_,_=run(q,patches,ph,pw,imgs.shape[2:],masks[0,0],fovs[0,0],dev)
v=fm>.5; y=(tr>.5).astype(np.uint8)[v]; a=fs[v]; b=qs[v]
def f1(s):
 p=s>=.5; tp=((p)&(y==1)).sum(); fp=((p)&(y==0)).sum(); fn=((~p)&(y==1)).sum(); return float(2*tp/max(2*tp+fp+fn,1))
print('SHAPES',fs.shape,tr.shape,fm.shape,'VALID',int(v.sum()),'TRUTH_POS_NEG',int((y==1).sum()),int((y==0).sum()))
print('FP32_SCORE',stats(fs),'INT4_SCORE',stats(qs),'PEARSON',float(pearsonr(a,b).statistic),'SPEARMAN',float(spearmanr(a,b).statistic))
print('FP32_F1_AUC',f1(a),float(roc_auc_score(y,a)),'INT4_F1_AUC',f1(b),float(roc_auc_score(y,b)),'INT4_FOREGROUND_RATIO',float((b>=.5).mean()),'FINITE',bool(np.isfinite(qs).all()))
for n,r in list(qo['layers'].items())[:3]:
 u=unpack(r); print('LAYER',n,'packed_dtype',str(r['weight_int4_packed'].dtype),'packed_shape',tuple(r['weight_int4_packed'].shape),'weight_shape',tuple(r['weight_shape']),'unpacked',tuple(u.shape),'scale',float(r['weight_scales'].min()),float(r['weight_scales'].max()),'zero_points',np.unique(r['weight_zero_points'].numpy()).tolist(),'qrange',int(u.min()),int(u.max()),'stats',stats(u.astype(np.float32)),'sat_frac',float(np.mean((u<=-7)|(u>=7))))
print('NIBBLE_ORDER','low_nibble_first; confirmed by tools/ptq_int4_baseline.py pack_signed_int4')
print('OUTPUT_WRITTEN False')

