import numpy as np

def count_classes(data):
    class_counts = {}
    for item in data:
        class_name = item[0]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    return class_counts

import torch

def jitter_point_cloud(pointcloud, sigma=0.01, clip=0.05):
    """
    Point cloud에 작은 노이즈를 추가하여 데이터 증강을 수행하는 함수입니다.
    
    :param pointcloud: 입력 포인트 클라우드 (Tensor)
    :param sigma: 노이즈 표준 편차 (기본값: 0.01)
    :param clip: 노이즈 절단 범위 (기본값: 0.05)
    :return: 노이즈가 추가된 포인트 클라우드
    """
    noise = torch.clamp(sigma * torch.randn(pointcloud.size()), -clip, clip)
    jittered_pointcloud = pointcloud + noise
    return jittered_pointcloud

import torch

def drop_random_points(pointcloud, drop_rate=0.2):
    """
    Point cloud에서 랜덤으로 포인트를 제거하는 함수입니다.
    
    :param pointcloud: 입력 포인트 클라우드 (Tensor)
    :param drop_rate: 드롭할 포인트의 비율 (기본값: 0.2)
    :return: 일부 포인트가 제거된 포인트 클라우드
    """
    num_points = pointcloud.size(0)
    num_drop = int(num_points * drop_rate)
    
    # 드롭할 포인트의 인덱스를 무작위로 선택
    drop_indices = torch.randperm(num_points)[:num_drop]
    remaining_indices = torch.ones(num_points, dtype=torch.bool)
    remaining_indices[drop_indices] = False
    
    dropped_pointcloud = pointcloud[remaining_indices]
    return dropped_pointcloud

import open3d as o3d
import numpy as np

def point_cloud_to_numpy(point_cloud):
    return np.asarray(point_cloud.points)

def numpy_to_point_cloud(numpy_array):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(numpy_array)
    return point_cloud

