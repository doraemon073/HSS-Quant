import argparse,hashlib,json,sys
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.extract_patches import load_data,my_PreProc,paint_border_overlap,extract_ordered_overlap,recompone_overlap
from lib.dataset import TestDataset
from models.UNetFamily import U_Net
from tools.group_trainable_bits import replace_conv_layers_with_groups
ROOT=Path(__file__).resolve().parents[1]; TEST=ROOT/'prepare_dataset/data_path_list/DRIVE/test.txt'
EXPECTED='c9c90f1b8b0ee85307090f83b8a2d4b99a615a99a13e06f3f232bcf4a411f774'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def binmask(x):
 x=np.asarray(x); return x if x.max()<=1 else x/255.0
def normalize_mask(mask,target_shape):
 x=np.asarray(mask); x=np.squeeze(x)
 if x.ndim!=2: raise AssertionError(('mask must be 2D',x.shape))
 h,w=target_shape
 if x.shape[0]<h or x.shape[1]<w: raise AssertionError(('mask smaller than target',x.shape,target_shape))
 return x[:h,:w]
def prepare_eval_arrays(scores,truth,fov,original_shape,crop_info):
 def two(x):
  x=np.asarray(x); x=np.squeeze(x)
  if x.ndim!=2: raise AssertionError(('expected 2D',x.shape))
  return x
 s=two(scores); t=two(truth); f=two(fov); h,w=original_shape
 if crop_info is not None:
  y0,y1,x0,x1=crop_info; s=s[y0:y1,x0:x1];t=t[y0:y1,x0:x1];f=f[y0:y1,x0:x1]
 if s.shape!=(h,w) or t.shape!=s.shape or f.shape!=s.shape: raise AssertionError((s.shape,t.shape,f.shape,original_shape,crop_info))
 s=s.astype(np.float32);t=(binmask(t)>0.5).astype(np.uint8);f=(binmask(f)>0.5).astype(np.float32)
 assert s.ndim==2 and t.ndim==2 and f.ndim==2 and s.shape==t.shape==f.shape
 assert np.isfinite(s).all() and np.isfinite(t).all() and np.isfinite(f).all()
 return s,t,f

def run_official(a,imgs,masks,fovs,padded,ph,pw,rows):
 outdir=(ROOT/a.output_dir).resolve()
 if outdir.exists(): raise FileExistsError(outdir)
 exp=ROOT/a.experiment; ck=torch.load(exp/'best_checkpoint.pt',map_location='cpu',weights_only=True)
 af=exp/'assignment.json'
 if not af.exists(): af=exp/'assignment_copy.json'
 if not af.exists(): raise FileNotFoundError('assignment.json or assignment_copy.json')
 ass=json.loads(af.read_text()); ar=ass.get('assignment',{})
 if len(ar)!=857 or ass.get('total_groups')!=857 or set(int(v) for v in ar.values())-{4,8}: raise RuntimeError('invalid assignment')
 if abs(float(ass['parameter_weighted_mean_bitwidth'])-4.269841064)>.01: raise RuntimeError('budget mismatch')
 base=torch.load(ROOT/'experiments/DRIVE_Unet_FP32_full_retry/best_model.pth',map_location='cpu',weights_only=True); model=U_Net(1,2); model.load_state_dict(base['net']); replace_conv_layers_with_groups(model,8); model.load_state_dict(ck['net']); model.cuda().eval()
 outdir.mkdir(); (outdir/'test_manifest_used.txt').write_text(TEST.read_text()); (outdir/'test_manifest_used.txt.sha256').write_text(sha(TEST)+'\n')
 patches=extract_ordered_overlap(padded,96,96,16,16); scores=[]; import time; started=time.perf_counter()
 with torch.no_grad():
  for j in range(0,len(patches),16): scores.append(torch.softmax(model(torch.from_numpy(patches[j:j+16]).float().cuda()), dim=1)[:,1].cpu().numpy())
 pred=recompone_overlap(np.expand_dims(np.concatenate(scores),1),ph,pw,16,16); from sklearn.metrics import roc_auc_score
 rows_out=[]; all_t=[];all_s=[]; import scipy.ndimage as ndi
 for i in range(20):
  sc,tr,fo=prepare_eval_arrays(pred[i,0],masks[i,0],fovs[i,0],imgs.shape[2:],(0,imgs.shape[2],0,imgs.shape[3])); valid=fo>.5; yt=tr[valid]; ys=sc[valid]
  if len(np.unique(yt))<2 or not np.isfinite(ys).all(): raise RuntimeError('invalid image metrics')
  auc=float(roc_auc_score(yt,ys)); yp=ys>=.5; tp=((yp)&(yt==1)).sum();fp=((yp)&(yt==0)).sum();fn=((~yp)&(yt==1)).sum();tn=((~yp)&(yt==0)).sum(); dice=float(2*tp/max(2*tp+fp+fn,1));iou=float(tp/max(tp+fp+fn,1));acc=float((tp+tn)/max(tp+tn+fp+fn,1));sens=float(tp/max(tp+fn,1));spec=float(tn/max(tn+fp,1)); rows_out.append({'image':i,'AUC':auc,'Dice/F1':dice,'IoU':iou,'Accuracy':acc,'Sensitivity':sens,'Specificity':spec});all_t.append(yt);all_s.append(ys)
 cropped_pred=[pred[ii,0,:imgs.shape[2],:imgs.shape[3]] for ii in range(20)]
 outimg=outdir/'result_img'; outimg.mkdir()
 for ii in range(20):
  vis=(np.clip(cropped_pred[ii],0,1)*255).astype(np.uint8)
  from PIL import Image
  Image.fromarray(vis).save(outimg/f'Result_{ii:02d}.png')
 from tools.structure_metrics_from_saved_predictions import analyze_prediction
 truth_crop=[normalize_mask(masks[ii,0],imgs.shape[2:]) for ii in range(20)]
 fov_crop=np.asarray([normalize_mask(fovs[ii,0],imgs.shape[2:]) for ii in range(20)])
 print('scores:',cropped_pred[0].shape)
 print('truth:',truth_crop[0].shape)
 print('fov:',fov_crop[0].shape)
 print('valid pixels:',int((binmask(fov_crop[0])>0.5).sum()))
 structure=analyze_prediction(cropped_pred, [binmask(x)>0.5 for x in truth_crop], fov_crop)
 keys=list(rows_out[0]);
 with (outdir/'comparison.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows_out)
 overall={k:float(np.mean([r[k] for r in rows_out])) for k in keys if k!='image'}; overall.update({'thin_vessel_recall':structure['thin_vessel_recall'],'thin_vessel_dice':structure['thin_vessel_dice'],'Skeleton_Dice':structure['skeleton_dice'],'clDice':structure['clDice'],'connected_components':structure['prediction_connected_components'],'uncovered_skeleton_fragments':structure['uncovered_skeleton_fragment_count']}); overall.update({'inference_passes':1,'test_seconds':time.perf_counter()-started,'gpu_peak_allocated_gib':float(torch.cuda.max_memory_allocated()/1024**3),'parameter_weighted_mean_bitwidth':float(ass['parameter_weighted_mean_bitwidth']),'theoretical_compression_ratio':float(ass.get('theoretical_compression_ratio',0))})
 (outdir/'performance.txt').write_text('\n'.join(f'{k}: {v}' for k,v in overall.items())+'\n'); np.save(outdir/'result.npy',np.asarray([np.concatenate(all_s),np.concatenate(all_t)])); (outdir/'quant_config.json').write_text(json.dumps(overall,indent=2)); (outdir/'README.md').write_text('Official one-pass FOV-only evaluation; fake quant; no training or calibration.\n'); (outdir/'inference_passes').write_text('1\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['sanity','smoke','official'],required=True);ap.add_argument('--experiment',default='experiments/DRIVE_Unet_GroupBits_stage1_full_seed2027_retry2');ap.add_argument('--output-dir');a=ap.parse_args()
 if sha(TEST)!=EXPECTED: raise RuntimeError('test manifest sha mismatch')
 rows=[x.strip().split() for x in TEST.read_text().splitlines() if x.strip()]; imgs,masks,fovs=load_data(str(TEST)); print('test_images',len(rows),'manifest_sha256',sha(TEST))
 proc=my_PreProc(imgs); padded=paint_border_overlap(proc,96,96,16,16); ph,pw=padded.shape[2:]; print('original_batch',imgs.shape,'padded_batch',padded.shape)
 for i in range(len(rows)):
  oh,ow=imgs.shape[2:]; crop=(0,oh,0,ow); scores=np.zeros((oh,ow),np.float32); truth=masks[i,0]; fov=fovs[i,0]; s,t,f=prepare_eval_arrays(scores,truth,fov,(oh,ow),crop); valid=f>.5; pos=int(((t>0)&valid).sum());neg=int(((t==0)&valid).sum());
  if pos<=0 or neg<=0: raise RuntimeError(f'image {i} single class pos={pos} neg={neg}')
  print(f'{i:02d} original={imgs[i].shape} padded={(1,)+padded[i].shape} final={s.shape} pos={pos} neg={neg} fov={int(valid.sum())}')
 print('totals',int(np.sum((masks[:,0]>127)&(fovs[:,0]>127))),int(np.sum((masks[:,0]<=127)&(fovs[:,0]>127))),int(np.sum(fovs[:,0]>127)))
 if a.mode=='sanity': return
 if a.mode=='official':
  if not a.output_dir: raise ValueError('--output-dir is required for official mode')
  run_official(a,imgs,masks,fovs,padded,ph,pw,rows); return
 base=torch.load(ROOT/'experiments/DRIVE_Unet_FP32_full_retry/best_model.pth',map_location='cpu',weights_only=True); ck=torch.load(ROOT/a.experiment/'best_checkpoint.pt',map_location='cpu',weights_only=True); model=U_Net(1,2);model.load_state_dict(base['net']);replace_conv_layers_with_groups(model,8);model.load_state_dict(ck['net']);model.cuda().eval(); patches=extract_ordered_overlap(padded[:1],96,96,16,16); out=[]
 with torch.no_grad():
  for j in range(0,len(patches),16): out.append(torch.softmax(model(torch.from_numpy(patches[j:j+16]).float().cuda()), dim=1)[:,1].cpu().numpy())
 pred=recompone_overlap(np.expand_dims(np.concatenate(out),1),ph,pw,16,16)[0,0];pred=pred[:imgs.shape[2],:imgs.shape[3]];s,t,f=prepare_eval_arrays(pred,masks[0,0],fovs[0,0],imgs.shape[2:],(0,imgs.shape[2],0,imgs.shape[3]));valid=f>.5;assert np.isfinite(s[valid]).all();print('SMOKE_PASS',{'scores_shape':s.shape,'truth_shape':t.shape,'fov_shape':f.shape,'output_shape':tuple(model(torch.from_numpy(patches[:1]).float().cuda()).shape),'pos':int((t[valid]>0).sum()),'neg':int((t[valid]==0).sum()),'test_manifest_read':True,'formal_results_written':False})
if __name__=='__main__':main()
