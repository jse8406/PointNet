import json
import numpy as np
import open3d as o3d
import os
from torch.utils.data import Dataset
import torch

# json 파일에서 annotation 읽기
def load_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    annotations = []
    for annotation in data["Annotation"]:
        class_name = annotation["class_name"]
        points = np.array(annotation["data"])
        annotations.append((class_name, points))
    return annotations

# pcd 파일에서 json 기준으로 bounding box 추출
def extract_bounding_box_points(pcd, bbox_points):
    min_bound = np.min(bbox_points, axis=0)
    max_bound = np.max(bbox_points, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pcd = pcd.crop(bbox)
    return cropped_pcd

# 디렉토리 탐색 및 파일 로드 (1000개씩)
def load_data_from_directory(base_path, phase, start_idx):
    data = []
    pcd_dir = os.path.join(base_path, phase, 'pcd')
    label_dir = os.path.join(base_path, phase, 'labels')
    
    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])[start_idx:start_idx+100]
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])[start_idx:start_idx+100]
    
    for pcd_file, label_file in zip(pcd_files, label_files):
        pcd_path = os.path.join(pcd_dir, pcd_file)
        label_path = os.path.join(label_dir, label_file)
        
        # PCD 파일 읽기
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        # JSON 파일 읽기
        annotations = load_annotations(label_path)
        for class_name, bbox_points in annotations:
            cropped_pcd = extract_bounding_box_points(pcd, bbox_points)
            data.append((class_name, cropped_pcd))
    
    return data

# 데이터셋 정의
class PointCloudDataset(Dataset):
    def __init__(self, data, num_points=1024):
        self.data = [(class_name, point_cloud) for class_name, point_cloud in data 
                     if np.asarray(point_cloud.points).shape[0] > 0]
        self.num_points = num_points
        self.label_map = {'car': 1, 'bus': 2, 'truck': 3, 'special vehicle': 4, 'motocycle': 5, 'bicycle': 6, 'personal mobility': 7, 'person': 8}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_name, point_cloud = self.data[idx] 
        points = np.asarray(point_cloud.points)
        label = self.label_map[class_name]
        
        if points.shape[0] > 0:
            if points.shape[0] > self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            else:
                indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            sampled_points = points[indices]
            return {'pointcloud': torch.tensor(sampled_points, dtype=torch.float32), 'category': torch.tensor(label, dtype=torch.long)}
        else:
            return
