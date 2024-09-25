from gradslam.structures.pointclouds import Pointclouds
import os
import open3d as o3d
import numpy as np

def create_markers_and_boxes():
    # 创建标记点
    markers = o3d.geometry.PointCloud()
    marker_points = np.array([
        [1, 1, 1],  # 标记点的坐标
        [2, 2, 2],
        [3, 3, 3]
    ])
    markers.points = o3d.utility.Vector3dVector(marker_points)
    markers.paint_uniform_color([1, 0, 0])  # 红色标记点

    # 创建边界框
    box1 = o3d.geometry.OrientedBoundingBox()
    box1.center = np.array([6.1,-0.6,-1.5])  # 边界框中心
    box1.extent = np.array([1.6,1.4,0.1])  # 边界框尺寸
    box1.color = [0, 1, 0]  # 绿色边界框
    
    box2 = o3d.geometry.OrientedBoundingBox()
    box2.center = np.array([3.7,1.0,-1.4])  # 边界框中心
    box2.extent = np.array([4.7,3.6,0.3])  # 边界框尺寸
    box2.color = [1, 0, 0]  # 红色边界框

    return box1, box2


box1, box2 = create_markers_and_boxes()

dir_to_save_map = os.path.join("/Users/zhangjie/Downloads/Replica", "room1", "rgb_cloud")
pointclouds = Pointclouds.load_pointcloud_from_h5(dir_to_save_map)
pcd = pointclouds.open3d(0)
geometries = [pcd, box1, box2]
o3d.visualization.draw_geometries(geometries)
# o3d.visualization.draw_geometries([pcd])