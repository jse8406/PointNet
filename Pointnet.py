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
        return {'pointcloud': torch.tensor(sampled_points, dtype=torch.float32),'category': torch.tensor(label, dtype=torch.long)}

# 데이터 로드 (전처리된 데이터 리스트 사용)

dataset = PointCloudDataset(data)
# print(dataset[1][0].shape) # 데이터셋 2차원 텐서(튜플)로 바뀜. 첫번째 인덱스는 포인트 클라우드, 두번째 인덱스는 라벨. 점 개수도 1024개로 통일
print(dataset[0])

from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=32, shuffle=True) # dataset을 batch_size만큼 묶어서 반환

# model 정의

import torch.nn as nn
import torch.nn.functional as F

## T-net

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        pool = nn.MaxPool1d(x.size(-1))(x)
        flat = nn.Flatten(1)(pool)
        
        x = F.relu(self.bn4(self.fc1(flat)))
        x = F.relu(self.bn5(self.fc2(x)))
        # x = self.fc3(x)
        
        iden = torch.eye(self.k).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        matrix = self.fc3(x).view(-1, self.k, self.k) + iden
        return matrix

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, k=9):
        super(PointNet, self).__init__()
        self.transform = Transform()
        # self.conv1 = nn.Conv1d(3, 64, 1)
        # self.conv2 = nn.Conv1d(64, 128, 1)
        # self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        xb, m3x3, m64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = torch.max(x, 2)[0]
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.bn5(self.fc2(x)))
        # x = self.dropout(x)
        # x = self.fc3(x)
        return self.logsoftmax(output), m3x3, m64x64

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pointnet = PointNet() 
pointnet.to(device)


# 모델 학습

import torch.optim as optim

# load if model exists
# pointnet.load_state_dict(torch.load('pointnet_model.pth'))

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pointnet.parameters(), lr=0.00025)

def train_model(model, train_loader, num_epochs=20):
    for epoch in range(num_epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
            print(inputs.size())
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')
    torch.save(model.state_dict(), 'pointnet_model.pth')
    

train_model(pointnet, train_loader)

