import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def unpack(rec):
 raw=rec['weight_int4_packed'].cpu().numpy().reshape(-1); lo=raw&15; hi=(raw>>4)&15
 vals=np.empty(raw.size*2,dtype=np.int8); vals[0::2]=np.where(lo>=8,lo-16,lo); vals[1::2]=np.where(hi>=8,hi-16,hi)
 return vals[:rec['weight_value_count']].reshape(tuple(rec['weight_shape']))
class QConv(nn.Module):
 def __init__(self,c,r):
  super().__init__(); self.conv=c; self.register_buffer('qi',torch.from_numpy(unpack(r)).to(torch.int8)); self.register_buffer('sc',r['weight_scales'].float()); self.register_buffer('zp',r['weight_zero_points'].int()); self.a=float(r['activation_scale']); self.az=int(r['activation_zero_point']); self.r=r
 def forward(self,x):
  x=torch.fake_quantize_per_tensor_affine(x,self.a,self.az,-8,7); p=self.r['conv_parameters']; w=(self.qi.float()-self.zp.float().view(-1,1,1,1))*self.sc.view(-1,1,1,1)
  return F.conv2d(x,w,self.conv.bias,tuple(p['stride']),tuple(p['padding']),tuple(p['dilation']),int(p['groups']))
def replace(m,L,p=''):
 for n,c in list(m.named_children()):
  full=f'{p}.{n}' if p else n
  if isinstance(c,nn.Conv2d):
   r=L[full]; setattr(m,n,QConv(c,r))
  else: replace(c,L,full)

