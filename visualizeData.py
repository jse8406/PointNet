import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# PCD 파일 로드
pcd_file = "./Data/training/pcd/E_DRG_230829_133_LR_316.pcd"  # 여기에 사용할 PCD 파일 경로 입력
pcd = o3d.io.read_point_cloud(pcd_file)

# NumPy 배열로 변환
points = np.asarray(pcd.points)

# 3D 시각화 (Open3D)
o3d.visualization.draw_geometries([pcd], window_name="PCD Visualization")

# # 2D 시각화 (Matplotlib)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='jet', alpha=0.6)

# ax.set_xlabel("X axis")
# ax.set_ylabel("Y axis")
# ax.set_zlabel("Z axis")
# ax.set_title("Point Cloud Visualization")

# plt.show()
