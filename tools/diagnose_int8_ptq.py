#!/usr/bin/env python3
"""Read-only one-image FP32 vs exported INT8 PTQ diagnostic."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.UNetFamily import U_Net
from lib.extract_patches import load_data, my_PreProc, paint_border_overlap, extract_ordered_overlap, recompone_overlap
from tools.evaluate_group_budget_official_test_v2 import prepare_eval_arrays

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "prepare_dataset/data_path_list/DRIVE/test.txt"
FP32 = ROOT / "experiments/DRIVE_Unet_FP32_full_retry/best_model.pth"
INT8 = ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/int8_weight_state_dict.pt"

def state_from(obj):
    if isinstance(obj, dict):
        for k in ("net", "state_dict", "model"):
            if isinstance(obj.get(k), dict):
                return obj[k]
    return obj

class ArtifactFakeQuantConv2d(nn.Module):
    def __init__(self, conv, rec):
        super().__init__()
        self.conv = conv
        self.register_buffer("weight_int8", rec["weight_int8"].to(torch.int8))
        self.register_buffer("weight_scales", rec["weight_scales"].float())
        self.register_buffer("weight_zero_points", rec["weight_zero_points"].to(torch.int32))
        self.activation_scale = float(rec["activation_scale"])
        self.activation_zero_point = int(rec["activation_zero_point"])
        self.rec = rec
    def forward(self, x):
        xq = torch.fake_quantize_per_tensor_affine(
            x, self.activation_scale, self.activation_zero_point, -128, 127)
        w = (self.weight_int8.float() - self.weight_zero_points.float().view(-1,1,1,1))
        w = w * self.weight_scales.float().view(-1,1,1,1)
        p = self.rec["conv_parameters"]
        return F.conv2d(xq, w, self.conv.bias, tuple(p["stride"]),
                        tuple(p["padding"]), tuple(p["dilation"]), int(p["groups"]))

def replace_artifact(module, layers, prefix=""):
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            if full not in layers:
                raise KeyError(f"missing exported INT8 layer: {full}")
            setattr(module, name, ArtifactFakeQuantConv2d(child, layers[full]))
        else:
            replace_artifact(child, layers, full)

def describe(a):
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std())}

def metrics(score, truth, fov, threshold):
    valid = fov > .5
    y = (truth > .5).astype(np.uint8)[valid]
    s = score[valid]
    p = s >= threshold
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    f1 = 2*tp / max(2*tp+fp+fn, 1)
    return f1, y, s

def run(model, patches, ph, pw, original_shape, truth, fov, device):
    model.eval()
    with torch.inference_mode():
        out = model(torch.from_numpy(patches).float().to(device))
        score = torch.softmax(out, dim=1)[:,1].cpu().numpy()
    pred = recompone_overlap(np.expand_dims(score,1), ph, pw, 16, 16)[0]
    sc,tr,fo = prepare_eval_arrays(pred, truth, fov, original_shape,
                                   (0, original_shape[0], 0, original_shape[1]))
    return out.cpu().numpy(), sc, tr, fo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fp_obj = torch.load(FP32, map_location="cpu", weights_only=False)
    q_obj = torch.load(INT8, map_location="cpu", weights_only=False)
    fp = U_Net(1,2); fp.load_state_dict(state_from(fp_obj), strict=True)
    q = U_Net(1,2); q.load_state_dict(state_from(fp_obj), strict=True)
    replace_artifact(q, q_obj["layers"])
    imgs,masks,fovs = load_data(str(TEST))
    img,mk,fv = imgs[:1], masks[0,0], fovs[0,0]
    padded = paint_border_overlap(my_PreProc(img), 96,96,16,16)
    ph,pw = padded.shape[2:]
    patches = extract_ordered_overlap(padded,96,96,16,16)
    fp.to(device); q.to(device)
    fo, fs, tr, fm = run(fp, patches, ph,pw, imgs.shape[2:], mk,fv,device)
    qo, qs, _, _ = run(q, patches, ph,pw, imgs.shape[2:], mk,fv,device)
    valid = fm > .5; yt=(tr>.5).astype(np.uint8)[valid]
    fscore=fs[valid]; qscore=qs[valid]
    print("CHECKPOINT_FP32_LOADED True"); print("CHECKPOINT_INT8_LOADED True")
    print("FP32_OUTPUT_SHAPE", tuple(fo.shape)); print("INT8_OUTPUT_SHAPE", tuple(qo.shape))
    print("FP32_LOGITS", describe(fo)); print("INT8_LOGITS", describe(qo))
    print("FP32_SCORE", describe(fs)); print("INT8_SCORE", describe(qs))
    print("PEARSON", float(pearsonr(fscore,qscore).statistic))
    print("SPEARMAN", float(spearmanr(fscore,qscore).statistic))
    print("FP32_FOREGROUND_RATIO", float((fscore>=.5).mean()))
    print("INT8_FOREGROUND_RATIO", float((qscore>=.5).mean()))
    print("TRUTH_POSITIVE", int((yt==1).sum()), "TRUTH_NEGATIVE", int((yt==0).sum()))
    for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        print(f"FP32_F1_{t}", metrics(fs,tr,fm,t)[0], f"INT8_F1_{t}", metrics(qs,tr,fm,t)[0])
    from sklearn.metrics import roc_auc_score
    print("FP32_AUC", float(roc_auc_score(yt,fscore))); print("INT8_AUC", float(roc_auc_score(yt,qscore)))
    print("SHAPES", fs.shape, tr.shape, fm.shape); print("FINITE", bool(np.isfinite(fs).all() and np.isfinite(qs).all()))
    for name,rec in list(q_obj["layers"].items())[:3]:
        wi=rec["weight_int8"].numpy(); print("LAYER",name,"qmin",int(wi.min()),"qmax",int(wi.max()),"scale_min",float(np.min(rec["weight_scales"].numpy())),"scale_max",float(np.max(rec["weight_scales"].numpy())),"sat_frac",float(np.mean((wi<=-127)|(wi>=127))))
    print("OUTPUT_WRITTEN False")
if __name__ == "__main__":
    main()

