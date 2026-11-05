import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.extract_patches import load_file_path_txt
from tools.structure_metrics_from_saved_predictions import load_test_ground_truth_and_fov,analyze_prediction
ROOT=Path(__file__).resolve().parents[1]; v5=ROOT/'experiments/DRIVE_Unet_INT4_PTQ_baseline/official_test_v5'; out=ROOT/'experiments/DRIVE_Unet_INT4_PTQ_baseline/official_test_v5_metrics_audit'
if out.exists(): raise FileExistsError(out)
result=np.load(v5/'result.npy',allow_pickle=False)
rows=load_file_path_txt(str(v5/'test_manifest_used.txt')); gt,fov=load_test_ground_truth_and_fov(rows)
assert result.shape==(20,584,565)
pred=result.astype(np.float32); truth=(gt[:,0]>0); masks=(fov[:,0]>0)
metrics=analyze_prediction(pred,truth,fov)
per=[]
for i in range(20):
 per.append({'image':i,**analyze_prediction(pred[i:i+1],truth[i:i+1],fov[i:i+1])})
out.mkdir()
with (out/'metrics_recomputed.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['metric','value','source']); [w.writerow([k,v,'historical_structure_metrics_from_saved_predictions']) for k,v in metrics.items()]
with (out/'per_image_metrics_recomputed.csv').open('w',newline='') as f:
 fields=list(per[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(per)
comparison={'thin_vessel_recall':'GT skeleton pixels with local width 2*EDT(GT vessel)<=2 px, dilated by one 8-neighbor pixel, intersected with GT vessel and FOV','thin_vessel_dice':'Dice over the thin-vessel target region used above, including prediction false positives in FOV','skeleton_dice':'2*intersection(prediction skeleton, truth skeleton)/(prediction skeleton pixels + truth skeleton pixels)','clDice':'harmonic mean of skeleton precision and skeleton sensitivity; with these definitions algebraically equals skeleton Dice','connected_components':'8-connected components of thresholded prediction restricted to FOV','uncovered_skeleton_fragments':'8-connected components of truth skeleton not covered by a one-pixel dilation of prediction, restricted to FOV','threshold':0.5,'fov_only':True,'result_shape':list(result.shape),'inference_run':False}
(out/'metric_definition_comparison.json').write_text(json.dumps(comparison,indent=2))
text='''# INT4 v5 structure-metric audit

This audit reads only official_test_v5/result.npy, the saved test manifest, and DRIVE truth/FOV files. No model inference, training, calibration, or checkpoint loading was performed.

## Recomputed historical metrics

'''
text+='\n'.join(f'- **{k}**: {v}' for k,v in metrics.items())+'\n\n'
text+='''## Findings

- The saved result is a full (20,584,565) score tensor, so no FOV-pixel flattening or reconstruction is needed for this audit.
- All metrics above use the historical structure_metrics_from_saved_predictions.py definitions and FOV restriction.
- Skeleton Dice and clDice are exactly equal here by algebra: the implemented clDice is the harmonic mean of skeleton precision and skeleton sensitivity, which reduces to 2I/(P+T), the same expression as Skeleton Dice. This equality is a property of the implementation, not evidence of independent metrics.
- Thin-vessel Dice can exceed thin-vessel recall because Dice includes false positives and has a different denominator; it is not required to be below recall.
- The high uncovered-fragment count is the historical fragment definition applied to the saved thresholded prediction. It counts connected components of uncovered truth skeleton after one-pixel prediction dilation; it is sensitive to skeleton fragmentation and should not be compared to earlier values unless the same prediction geometry, threshold, skeletonization, and FOV procedure were used.
- The v5 runner original structure values were not accepted as historical-comparable until this audit; the recomputed values are the values to use for comparison.

## Disposition

Basic pixel metrics in v5 are supported by finite full-resolution scores, aligned truth/FOV, and the saved manifest. Structure metrics are usable only with the recomputed historical definitions above. Do not mix the runner pre-audit structure columns with historical reports. No new inference is required unless exact paper comparability requires regenerating a result with a corrected runner.
'''
(out/'audit_report.md').write_text(text)
print('AUDIT_METRICS',metrics)

