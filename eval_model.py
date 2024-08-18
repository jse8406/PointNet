from sklearn.metrics import average_precision_score
import numpy as np
import torch

def calculate_iou(pred, target, num_classes):
    iou_list = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for cls in range(1, num_classes):  # Assuming 0 is background class
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            # iou_list.append(float('nan'))  # Only count if union is not zero
            continue
        else:
            iou_list.append(intersection / union)
    return np.nanmean(iou_list)

def calculate_map(all_preds, all_labels, num_classes):
    # One-hot encode labels and predictions for AP calculation
    all_labels_one_hot = torch.zeros(all_labels.size(0), num_classes)
    all_preds_one_hot = torch.zeros(all_preds.size(0), num_classes)

    for c in range(num_classes):
        all_labels_one_hot[:, c] = (all_labels == c).float()
        all_preds_one_hot[:, c] = (all_preds == c).float()

    # Calculate AP for each class
    ap_per_class = []
    for c in range(num_classes):
        if torch.sum(all_labels_one_hot[:, c]) == 0:  # 해당 클래스가 없으면 생략
            print(c)
            continue
        ap = average_precision_score(all_labels_one_hot[:, c].cpu().numpy(), all_preds_one_hot[:, c].cpu().numpy())
        ap_per_class.append(max(ap, 0))  # AP가 음수인 경우 0으로 처리
    print(ap_per_class)
    # Mean of all classes' AP
    if len(ap_per_class) == 0:
        return 0.0
    mAP = sum(ap_per_class) / len(ap_per_class)
    return mAP
