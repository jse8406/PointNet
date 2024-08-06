# import numpy as np
# import torch


# def calculate_iou(box1, box2):
#     # box: [x_min, y_min, z_min, x_max, y_max, z_max]
#     x_min_inter = max(box1[0], box2[0])
#     y_min_inter = max(box1[1], box2[1])
#     z_min_inter = max(box1[2], box2[2])
#     x_max_inter = min(box1[3], box2[3])
#     y_max_inter = min(box1[4], box2[4])
#     z_max_inter = min(box1[5], box2[5])
    
#     inter_volume = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter) * max(0, z_max_inter - z_min_inter)
#     box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
#     box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    
#     iou = inter_volume / (box1_volume + box2_volume - inter_volume)
#     return iou


# def calculate_precision_recall(true_boxes, pred_boxes, iou_threshold=0.5):
#     true_boxes_detected = [False] * len(true_boxes)
#     pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[1], reverse=True)  # sort by confidence

#     tp = 0
#     fp = 0
#     precision = []
#     recall = []
    
#     for pred_box in pred_boxes_sorted:
#         pred_box_coords = pred_box[0]
#         pred_box_conf = pred_box[1]
        
#         best_iou = 0
#         best_iou_idx = -1
#         for i, true_box in enumerate(true_boxes):
#             if not true_boxes_detected[i]:
#                 iou = calculate_iou(pred_box_coords, true_box)
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_iou_idx = i
                
#         if best_iou >= iou_threshold:
#             true_boxes_detected[best_iou_idx] = True
#             tp += 1
#         else:
#             fp += 1
        
#         precision.append(tp / (tp + fp))
#         recall.append(tp / len(true_boxes))
    
#     return precision, recall

# def calculate_ap(precision, recall):
#     precision = [0] + precision + [0]
#     recall = [0] + recall + [1]

#     for i in range(len(precision) - 1, 0, -1):
#         precision[i - 1] = max(precision[i - 1], precision[i])

#     ap = 0
#     for i in range(1, len(precision)):
#         ap += (recall[i] - recall[i - 1]) * precision[i]

#     return ap

# def calculate_map(true_boxes_list, pred_boxes_list, iou_threshold=0.5):
#     ap_list = []
#     for true_boxes, pred_boxes in zip(true_boxes_list, pred_boxes_list):
#         precision, recall = calculate_precision_recall(true_boxes, pred_boxes, iou_threshold)
#         ap = calculate_ap(precision, recall)
#         ap_list.append(ap)
#     return np.mean(ap_list)

# def test_model_with_map(model, test_loader, iou_threshold=0.5):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     true_boxes_list = []
#     pred_boxes_list = []
    
#     with torch.no_grad():
#         for data in test_loader:
#             inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
#             outputs, __, __ = model(inputs.transpose(1, 2))
#             _, predicted = torch.max(outputs.data, 1)
            
#             # Add true and predicted boxes to lists for mAP calculation
#             true_boxes_list.append(labels.cpu().numpy())
#             pred_boxes_list.append(predicted.cpu().numpy())
    
#     mAP = calculate_map(true_boxes_list, pred_boxes_list, iou_threshold=iou_threshold)
#     print(f'Test mAP: {mAP:.2f}')
