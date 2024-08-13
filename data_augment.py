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


# 랜덤 노이즈 추가
def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
    jitter = np.clip(sigma * np.random.randn(*point_cloud.shape), -clip, clip)
    point_cloud += jitter
    return point_cloud


# 랜덤 드롭
def drop_random_points(point_cloud, drop_rate=0.1):
    num_points = point_cloud
    drop_indices = np.random.choice(num_points, int(num_points * drop_rate), replace=False)
    return np.delete(point_cloud, drop_indices, axis=0)

import open3d as o3d
import numpy as np

def point_cloud_to_numpy(point_cloud):
    return np.asarray(point_cloud.points)

def numpy_to_point_cloud(numpy_array):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(numpy_array)
    return point_cloud

