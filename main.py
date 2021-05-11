import numpy as np
import open3d as o3d

import util
import cmpl.shinv


"""
Settings
"""

CAN_MESH_PATH = 'can.obj'


"""
Main Process
"""

# load original mesh
can_orig_mesh = util.load_mesh(CAN_MESH_PATH)

# crop partial mesh
can_part_mesh = util.crop_mesh(can_orig_mesh)
can_part_pcd = util.convert_mesh2pcd(can_part_mesh)

# apply completion process
can_part_pcd_points = np.asarray(can_part_pcd.points, dtype=np.float32)
can_comp_pcd_points = cmpl.shinv.run(can_part_pcd_points)

# create point cloud object
can_comp_pcd = o3d.geometry.PointCloud()
can_comp_pcd.points = o3d.utility.Vector3dVector(can_comp_pcd_points)

# get mesh results from point clouds
can_part_result = util.convert_pcd2mesh(can_part_pcd)
can_comp_result = util.convert_pcd2mesh(can_comp_pcd)

# visualize results
util.visualize_mesh(can_part_result)
util.visualize_mesh(can_comp_result)
