import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.UNetFamily import U_Net
from tools.group_trainable_bits import replace_conv_layers_with_groups
from lib.extract_patches import get_data_test_overlap, recompone_overlap, load_file_path_txt
from tools.structure_metrics_from_saved_predictions import analyze_prediction

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def set_assignment(wrappers, assignment):
    with torch.no_grad():
        for name, wrapper in wrappers.items():
            for gid in range(wrapper.num_groups):
                key = f'{name}:{gid}'
                if key not in assignment:
                    raise ValueError(f'missing assignment {key}')
                wrapper.bit_logit.data[gid] = 20.0 if int(assignment[key]) == 8 else -20.0
    for wrapper in wrappers.values():
        wrapper.bit_logit.requires_grad_(False)

def pixel_metrics(scores, truth, fov):
    valid = fov > 0.5
    y = (truth > 0.5).astype(np.uint8)
    s = np.asarray(scores, dtype=np.float32)
    yv, sv = y[valid], s[valid]
    pred = sv >= 0.5
    tp = int(np.sum(pred & (yv == 1))); fp = int(np.sum(pred & (yv == 0)))
    fn = int(np.sum((~pred) & (yv == 1))); tn = int(np.sum((~pred) & (yv == 0)))
    auc = float(roc_auc_score(yv, sv)) if len(np.unique(yv)) == 2 else float('nan')
    return {
        'AUC': auc,
        'Dice/F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        'IoU': tp/(tp+fp+fn) if tp+fp+fn else 0.0,
        'Accuracy': (tp+tn)/(tp+tn+fp+fn),
        'Sensitivity': tp/(tp+fn) if tp+fn else 0.0,
        'Specificity': tn/(tn+fp) if tn+fp else 0.0,
        'FOV pixels': int(valid.sum()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', choices=['fp32', 'learned_g8', 'structural'], required=True)
    ap.add_argument('--experiment', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--test-manifest', default='/data/xpp/data/CHASE_DB1-main/CHASE_DB1-main/prepared_seed2026/test.txt')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = (ROOT / args.output_dir).resolve(); exp = (ROOT / args.experiment).resolve()
    manifest = Path(args.test_manifest).resolve()
    if out.exists(): raise SystemExit(f'OUTPUT_EXISTS: {out}')
    image_paths, label_paths, fov_paths = load_file_path_txt(str(manifest))
    if len(image_paths) != 8: raise ValueError(f'expected 8 CHASE test images, got {len(image_paths)}')
    out.mkdir(parents=False); (out/'result_img').mkdir()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = U_Net(1, 2)
    assignment = None
    if args.model_type == 'fp32':
        checkpoint_path = exp / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['net'], strict=True)
        wrappers = {}
    else:
        checkpoint_path = exp / 'best_checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        wrappers = replace_conv_layers_with_groups(model, 8)
        state = checkpoint.get('net', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state, strict=True)
        assignment = json.loads((exp/'assignment.json').read_text())['assignment']
        set_assignment(wrappers, assignment)
    model.to(device).eval()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    patches, raw_images, masks, fovs, new_h, new_w = get_data_test_overlap(str(manifest), 96, 96, 16, 16)
    outputs = []
    with torch.no_grad():
        for i in range(0, len(patches), 64):
            batch = torch.from_numpy(patches[i:i+64]).float().to(device)
            outputs.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    pred_patches = np.expand_dims(np.concatenate(outputs, axis=0), 1)
    full = recompone_overlap(pred_patches, new_h, new_w, 16, 16)
    height, width = raw_images.shape[2], raw_images.shape[3]
    scores = full[:, 0, :height, :width].astype(np.float32)
    truth = (masks[:, 0] > 0.5).astype(bool)
    fov = (fovs[:, 0] > 0.5).astype(bool)
    if scores.shape != truth.shape or truth.shape != fov.shape:
        raise ValueError(f'shape mismatch: {scores.shape} {truth.shape} {fov.shape}')
    if not np.isfinite(scores).all(): raise ValueError('NaN/Inf in scores')
    for i, image_path in enumerate(image_paths):
        valid = fov[i]
        if not valid.any() or not truth[i][valid].any() or (~truth[i][valid]).all():
            raise ValueError(f'FOV class check failed at {image_path}')
        name = Path(image_path).stem
        Image.fromarray((scores[i] >= 0.5).astype(np.uint8)*255).save(out/'result_img'/(name+'.png'))
    metrics = pixel_metrics(scores, truth, fov)
    structure = analyze_prediction(scores, truth, fov[:, None, :, :])
    metrics.update({'thin-vessel recall': structure['thin_vessel_recall'], 'thin-vessel Dice': structure['thin_vessel_dice'], 'Skeleton Dice': structure['skeleton_dice'], 'clDice': structure['clDice'], 'connected components': structure['prediction_connected_components'], 'uncovered skeleton fragments': structure['uncovered_skeleton_fragment_count']})
    np.save(out/'result.npy', np.stack([truth.astype(np.uint8), scores], axis=0))
    np.save(out/'scores.npy', scores); np.save(out/'truth.npy', truth); np.save(out/'fov.npy', fov)
    (out/'test_manifest_used.txt').write_text(manifest.read_text()); (out/'test_manifest_used.txt.sha256').write_text(sha(manifest)+'  test_manifest_used.txt\n'); (out/'inference_passes').write_text('1\n')
    with (out/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics)); w.writeheader(); w.writerow(metrics)
    with (out/'comparison.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric','value']); w.writerows(metrics.items())
    with (out/'per_image_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id','AUC','Dice/F1','IoU','FOV pixels']); w.writeheader()
        for i, image_path in enumerate(image_paths):
            m = pixel_metrics(scores[i], truth[i], fov[i]); w.writerow({'image_id': Path(image_path).stem, **{k:m[k] for k in ['AUC','Dice/F1','IoU','FOV pixels']}})
    config = {'dataset':'CHASE_DB1','model_type':args.model_type,'checkpoint':str(checkpoint_path),'test_manifest':str(manifest),'test_manifest_sha256':sha(manifest),'image_count':len(image_paths),'original_shape':[height,width],'patch_size':[96,96],'stride':[16,16],'fov_only':True,'foreground':'softmax class 1','assignment':assignment,'inference_passes':1,'elapsed_seconds':time.perf_counter()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/(1024**3) if torch.cuda.is_available() else 'unavailable'}
    (out/'quant_config.json').write_text(json.dumps(config, indent=2)); (out/'README.md').write_text('CHASEDB1 official evaluation; FOV-only; class-1 softmax; original-size crop; no resize or broadcasting.\n\n'+json.dumps(config, indent=2))
    (out/'performance.txt').write_text('\n'.join(f'{k}: {v}' for k,v in metrics.items())+'\n')
    print(json.dumps({'output':str(out),'metrics':metrics,'png_count':len(list((out/'result_img').glob('*.png'))),'elapsed_seconds':config['elapsed_seconds']}, indent=2))

if __name__ == '__main__': main()
