import argparse,csv,hashlib,json,time,sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy import ndimage
from skimage.morphology import skeletonize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.extract_patches import load_data,my_PreProc,paint_border_overlap,extract_ordered_overlap,recompone_overlap
from tools.evaluate_group_budget_official_test_v2 import prepare_eval_arrays
from tools.diagnose_int8_ptq import replace_artifact
from tools.int4_artifact import replace as replace_int4
from models.UNetFamily import U_Net
ROOT=Path(__file__).resolve().parents[1]; TEST=ROOT/'prepare_dataset/data_path_list/DRIVE/test.txt'; FP32=ROOT/'experiments/DRIVE_Unet_FP32_full_retry/best_model.pth'
def save_prediction_png(scores_image, output_path):
 scores_image=np.asarray(scores_image)
 assert scores_image.ndim==2 and scores_image.shape==(584,565)
 binary_prediction=scores_image>=0.5
 assert binary_prediction.shape==(584,565)
 Image.fromarray(binary_prediction.astype(np.uint8)*255).save(output_path)
def state(o):
 if isinstance(o,dict):
  for k in ('net','state_dict','model'):
   if isinstance(o.get(k),dict): return o[k]
 return o
def build_model(kind,ck,device):
 o=torch.load(ck,map_location='cpu',weights_only=False); m=U_Net(1,2)
 if kind=='fp32': m.load_state_dict(state(o),strict=True)
 else:
  base=torch.load(FP32,map_location='cpu',weights_only=False); m.load_state_dict(state(base),strict=True)
  if 'layers' not in o: raise RuntimeError('quant artifact missing layers')
  if kind=='int8': replace_artifact(m,o['layers'])
  elif kind=='int4': replace_int4(m,o['layers'])
  else: raise ValueError(kind)
 return m.to(device).eval()
def struct(pred,y,f):
 v=f>.5; p=pred&v; t=y&v; skt=skeletonize(t); skp=skeletonize(p)
 thin=(skt)
 tp=(pred[thin]&t[thin]).sum(); rec=float(tp/max(skt.sum(),1)); td=float(2*tp/max(pred[thin].sum()+skt.sum(),1))
 inter=(skp&skt).sum(); sd=float(2*inter/max(skp.sum()+skt.sum(),1))
 def prec(a,b): return float((a&b).sum()/max(a.sum(),1))
 cl=float(2*prec(skp,skt)*prec(skt,skp)/max(prec(skp,skt)+prec(skt,skp),1e-12))
 cc=int(ndimage.label(p)[1]); labs,n=ndimage.label(skt); unc=0
 for k in range(1,n+1):
  z=labs==k
  if not (p&z).any(): unc+=1
 return rec,td,sd,cl,cc,int(unc)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-type',choices=['fp32','int8','int4'],required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args()
 out=(ROOT/a.output_dir).resolve()
 if out.exists(): raise FileExistsError(out)
 ck=(ROOT/a.checkpoint).resolve()
 if ck.is_dir(): ck=ck/({'int8':'int8_weight_state_dict.pt','int4':'int4_weight_state_dict.pt'}[a.model_type])
 if not ck.exists(): raise FileNotFoundError(ck)
 imgs,masks,test_fovs_array=load_data(str(TEST)); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); m=build_model(a.model_type,ck,dev)
 assert not isinstance(test_fovs_array,list)
 padded=paint_border_overlap(my_PreProc(imgs),96,96,16,16); ph,pw=padded.shape[2:]; patches=extract_ordered_overlap(padded,96,96,16,16)
 if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
 start=time.perf_counter(); zs=[]
 with torch.no_grad():
  for j in range(0,len(patches),16): zs.append(torch.softmax(m(torch.from_numpy(patches[j:j+16]).float().to(dev)),1)[:,1].cpu().numpy())
 scores=recompone_overlap(np.expand_dims(np.concatenate(zs),1),ph,pw,16,16)[:,0]; rows=[]; truths=[]; fov_results=[]; out.mkdir(parents=True); (out/'result_img').mkdir()
 for i in range(len(imgs)):
  sc,tr,fo=prepare_eval_arrays(scores[i],masks[i,0],test_fovs_array[i,0],imgs.shape[2:],(0,imgs.shape[2],0,imgs.shape[3])); assert sc.shape==tr.shape==fo.shape==(584,565)
  v=fo>.5; y=(tr>.5).astype(np.uint8)[v]; s=sc[v]
  if np.unique(y).size<2 or not np.isfinite(s).all(): raise RuntimeError(f'invalid image {i}')
  b=s>=.5; tp=((b)&(y==1)).sum(); fp=((b)&(y==0)).sum(); fn=((~b)&(y==1)).sum(); tn=((~b)&(y==0)).sum()
  rec,td,sd,cl,cc,unc=struct(sc>=.5,tr>.5,fo)
  rows.append({'image':i,'AUC':float(roc_auc_score(y,s)),'Dice/F1':float(2*tp/max(2*tp+fp+fn,1)),'IoU':float(tp/max(tp+fp+fn,1)),'Accuracy':float((tp+tn)/max(tp+tn+fp+fn,1)),'Sensitivity':float(tp/max(tp+fn,1)),'Specificity':float(tn/max(tn+fp,1)),'thin-vessel recall':rec,'thin-vessel Dice':td,'Skeleton Dice':sd,'clDice':cl,'connected components':cc,'uncovered skeleton fragments':unc,'FOV pixels':int(v.sum())})
  assert sc.ndim==2 and tr.ndim==2 and fo.ndim==2 and sc.shape==tr.shape==fo.shape==(584,565)
  truths.append(tr); fov_results.append(fo); save_prediction_png(sc, out/'result_img'/f'Result_{i:02d}.png')
 np.save(out/'result.npy',scores)
 fields=list(rows[0])
 with (out/'per_image_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 overall={k:(float(np.mean([r[k] for r in rows])) if k not in ('image','connected components','uncovered skeleton fragments','FOV pixels') else int(np.sum([r[k] for r in rows]))) for k in fields}; overall.update({'model_type':a.model_type,'checkpoint':str(ck),'inference_passes':1,'test_seconds':time.perf_counter()-start,'gpu_peak_memory_bytes':int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else 'unavailable'})
 for name in ('metrics.csv','comparison.csv'):
  with (out/name).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(overall)); w.writeheader(); w.writerow(overall)
 (out/'performance.txt').write_text('\n'.join(f'{k}: {v}' for k,v in overall.items())+'\nFOV-only metrics: yes\nFake quant: yes\n')
 (out/'inference_passes').write_text('1\n'); (out/'test_manifest_used.txt').write_bytes(TEST.read_bytes()); (out/'test_manifest_used.txt.sha256').write_text(hashlib.sha256(TEST.read_bytes()).hexdigest()+'\n')
 (out/'quant_config.json').write_text(json.dumps({'model_type':a.model_type,'artifact':str(ck),'fake_quant':True,'foreground':'softmax class 1','theoretical_metrics_only':True},indent=2))
 (out/'README.md').write_text('Official DRIVE test; one inference pass; FOV-only; no training/calibration.\n')
if __name__=='__main__': main()

