import open3d as o3d
import json
import numpy as np


# PCD 파일 읽기

pcd = o3d.io.read_point_cloud("Data/test/pcd/E_DRG_230829_133_LR_159.pcd")

        ## 포인트 클라우드 시각화
        ## o3d.visualization.draw_geometries([pcd])

# json 파일 읽기

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

annotations = load_annotations("Data/test/labels/E_DRG_230829_133_LR_159.json")



data = []
for class_name, bbox_points in annotations:
    cropped_pcd = extract_bounding_box_points(pcd, bbox_points)
    data.append((class_name, cropped_pcd))

# data에 라벨과 포인트 클라우드가 저장되어 있음

import torch
from torch.utils.data import Dataset

# pointnet에 학습시킬 수 있는 형태로 데이터셋 정의(텐서로 변환)

class PointCloudDataset(Dataset):
    def __init__(self, data, num_points=1024):
        self.data = data
        self.num_points = num_points
        # 객체 종류 총 8개
        self.label_map = {'car': 1, 'bus': 2, 'truck': 3, 'special vehicle': 4, 'motocycle': 5, 'bicycle': 6, 'personal mobility': 7, 'person': 8}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_name, point_cloud = self.data[idx]
        points = np.asarray(point_cloud.points)
        label = self.label_map[class_name]
        
        # 포인트 샘플링 (필요한 경우 패딩)
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
        
        sampled_points = points[indices]
        return torch.tensor(sampled_points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 데이터 로드 (전처리된 데이터 리스트 사용)

dataset = PointCloudDataset(data)



from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=32, shuffle=True) # dataset을 batch_size만큼 묶어서 반환

# model 정의

import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, k=9):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

pointnet = PointNet(k=9)  # 두 개의 클래스로 분류


# 모델 학습

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pointnet.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.permute(0, 2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')
    torch.save(model.state_dict(), 'pointnet_model.pth')
    

train_model(pointnet, train_loader)

