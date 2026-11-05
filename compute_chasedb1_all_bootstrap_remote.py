import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data/xpp/VesselSeg-Pytorch-master/VesselSeg-Pytorch-master")
from tools.structure_metrics_from_saved_predictions import analyze_prediction

ROOT = Path("/data/xpp/VesselSeg-Pytorch-master/VesselSeg-Pytorch-master")
FOV_ROOT = Path("/data/xpp/UPEN_Mamba/results_chasedb1_fov_official_mamba_converged")
FOV = np.load(FOV_ROOT / "fov.npy") > 0.5
TRUTH = np.load(FOV_ROOT / "truth.npy") > 0.5
METHODS = {
    "FP32 U-Net": "experiments/CHASEDB1_Unet_FP32_seed2026_test/result.npy",
    "Learned-G8 QAT": "experiments/CHASEDB1_GroupBits_QAT_learned_seed2026_run2_test_retry2/result.npy",
    "Structural QAT": "experiments/CHASEDB1_GroupBits_QAT_structural_seed2026_test_final/result.npy",
    "Gradient/Taylor QAT": "experiments/CHASEDB1_GroupBits_gradient_taylor_QAT_seed2026_test_retry/result.npy",
    "Hessian/HAWQ QAT": "experiments/CHASEDB1_GroupBits_hessian_hawq_QAT_seed2026_test/result.npy",
    "Structural QAT thin4": "experiments/CHASEDB1_GroupBits_QAT_structural_thin4_control_seed2026_test/result.npy",
    "Structural QAT thin8": "experiments/CHASEDB1_GroupBits_QAT_structural_thin8_seed2026_test_retry2/result.npy",
}
NAMES = ["AUC", "Dice", "IoU", "Sensitivity", "ThinRecall", "SkeletonDice", "clDice"]
rng = np.random.default_rng(20261028)
all_out = {"n_images": int(FOV.shape[0]), "bootstrap_resamples": 10000, "seed": 20261028, "methods": {}}

for name, rel in METHODS.items():
    saved = np.load(ROOT / rel, allow_pickle=False)
    saved_truth = saved[0] > 0.5
    scores = saved[1].astype(np.float32)
    if not np.array_equal(saved_truth, TRUTH):
        raise RuntimeError(f"truth mismatch for {name}")
    per = []
    for i in range(scores.shape[0]):
        valid = FOV[i]
        auc = roc_auc_score(TRUTH[i][valid].astype(np.uint8), scores[i][valid])
        m = analyze_prediction(scores[i:i+1], TRUTH[i:i+1], FOV[i:i+1, None, :, :])
        per.append([auc, m["overall_dice"], m["overall_iou"], m["vessel_sensitivity_recall"], m["thin_vessel_recall"], m["skeleton_dice"], m["clDice"]])
    per = np.asarray(per, dtype=float)
    boot = np.empty((10000, per.shape[1]), dtype=float)
    for b in range(len(boot)):
        boot[b] = np.mean(per[rng.integers(0, len(per), size=len(per))], axis=0)
    rec = {"image_level_mean": {}, "bootstrap_95ci": {}, "per_image": {}}
    for j, metric in enumerate(NAMES):
        rec["image_level_mean"][metric] = float(np.mean(per[:, j]))
        rec["bootstrap_95ci"][metric] = [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
        rec["per_image"][metric] = [float(x) for x in per[:, j]]
    all_out["methods"][name] = rec

out_path = ROOT / "experiments/CHASEDB1_bootstrap_all_methods.json"
out_path.write_text(json.dumps(all_out, indent=2))
print(json.dumps(all_out, indent=2))
