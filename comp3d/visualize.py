import sys
import h5py
import numpy as np
import open3d as o3d

h5path = sys.argv[1]
h5file = h5py.File(h5path, 'r')

for key1 in h5file.keys():
    col = h5file[key1]
    for key2 in col.keys():
        pts = np.array(h5file[key1][key2]).squeeze(0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw_geometries([pcd])