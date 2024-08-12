from sklearn.metrics import average_precision_score
import numpy as np

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

def calculate_map(pred, target, num_classes):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    ap_list = []
    for cls in range(1, num_classes):  # Assuming 0 is background class
        pred_binary = (pred == cls).astype(int)
        target_binary = (target == cls).astype(int)
        ap = average_precision_score(target_binary, pred_binary)
        ap_list.append(ap)
    return np.mean(ap_list)
