import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm

pred_dir = 'id_/eval/predicted_masks'
gt_dir = 'id_/eval/ground_truth_masks'

files = [f for f in os.listdir(gt_dir) if f.endswith('.png') or f.endswith('.jpg')]

precisions, recalls, f1s, ious = [], [], [], []
matched_count = 0

for file in tqdm(files, desc="Evaluating"):
    file_base = os.path.splitext(file)[0]
    pred_path = os.path.join(pred_dir, f"pred_{file_base}.png")
    gt_path = os.path.join(gt_dir, file)

    if not os.path.exists(pred_path):
        print(f"[Missing] Predicted mask not found: {pred_path}")
        continue

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        print(f"[Error] Could not read: {file}")
        continue

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

    pred_flat = pred_bin.flatten()
    gt_flat = gt_bin.flatten()

    precisions.append(precision_score(gt_flat, pred_flat, zero_division=0))
    recalls.append(recall_score(gt_flat, pred_flat, zero_division=0))
    f1s.append(f1_score(gt_flat, pred_flat, zero_division=0))
    ious.append(jaccard_score(gt_flat, pred_flat, zero_division=0))
    matched_count += 1

print("\nEvaluation Summary:")
print(f"Matched pairs evaluated: {matched_count} / {len(files)}")
if matched_count == 0:
    print(" No valid ground truth & predicted mask pairs were matched.")
else:
    print(f" Precision: {np.mean(precisions):.4f}")
    print(f" Recall:    {np.mean(recalls):.4f}")
    print(f" F1-Score:  {np.mean(f1s):.4f}")
    print(f" IoU:       {np.mean(ious):.4f}")
