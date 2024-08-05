import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import json

# T-Net definition
class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
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
        self.k = k

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = torch.eye(self.k).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.k, self.k) + iden
        return x

# PointNet definition
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.tnet1 = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.tnet2 = TNet(k=64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1)
        trans = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        trans_feat = self.tnet2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, trans_feat

# PointNetLoss definition
class PointNetLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target, trans_feat):
        loss = F.cross_entropy(pred, target)
        mat_diff_loss = self.mat_diff_loss(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

    def mat_diff_loss(self, trans):
        batch_size = trans.size(0)
        k = trans.size(1)
        I = torch.eye(k, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
        diff = trans @ trans.transpose(2, 1) - I
        return torch.mean(torch.sum(diff ** 2, dim=(1, 2)))

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
# Load PCD and JSON files
def load_point_cloud(pcd_file):
    return o3d.io.read_point_cloud(pcd_file)

def load_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    annotations = []
    for annotation in data["Annotation"]:
        class_name = annotation["class_name"]
        points = np.array(annotation["data"])
        annotations.append((class_name, points))
    return annotations

def extract_bounding_box_points(pcd, bbox_points):
    min_bound = np.min(bbox_points, axis=0)
    max_bound = np.max(bbox_points, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)

# Prepare data
pcd_file = "Data/test/pcd/E_DRG_230829_133_LR_159.pcd"
json_file = "Data/test/labels/E_DRG_230829_133_LR_159.json"
pcd = load_point_cloud(pcd_file)
annotations = load_annotations(json_file)

objects = []
for class_name, bbox_points in annotations:
    cropped_pcd = extract_bounding_box_points(pcd, bbox_points)
    objects.append((class_name, cropped_pcd))

# Load data
dataset = PointCloudDataset(objects)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model and loss function definition
num_classes = 9
model = PointNet(num_classes)
criterion = PointNetLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
def train_model(model, train_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, trans_feat = model(inputs)
            loss = criterion(outputs, labels, trans_feat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

train_model(model, train_loader)
