import os
import numpy as np

import util
import config

"""
Create Point Cloud Data
"""

# initialize data list
complete_pcds = []
incomplete_pcds = []
labels = []

# loop over classes
class_size = len(config.CLASS_IDS)
for class_index, class_id in enumerate(config.CLASS_IDS):

    # skip if class ID is empty
    if not class_id:
        continue

    # set class path
    class_path = os.path.join(config.DATASET_PATH, class_id)

    # loop over models
    model_ids = os.listdir(class_path)
    model_size = len(model_ids)
    for model_index, model_id in enumerate(model_ids):

        # set model path
        model_path = os.path.join(class_path, model_id)

        # set mesh and point cloud directory path
        mesh_path = os.path.join(model_path, config.MESH_DIRECTORY)
        pcd_path  = os.path.join(model_path, config.PCD_DIRECTORY)

        # create point cloud directory
        if not os.path.exists(pcd_path):
            os.mkdir(pcd_path)

        # set mesh and point cloud file path
        triangle_mesh_path   = os.path.join(mesh_path, config.MESH_FILENAME_TRI)
        complete_mesh_path   = os.path.join(mesh_path, config.MESH_FILENAME_ORG)
        complete_pcd_path    = os.path.join(pcd_path , config.PCD_FILENAME_COMPLETE)

        # print info message
        print("[{}/{}][{}/{}] {}".format(class_index + 1, class_size,
                                         model_index + 1, model_size,
                                         model_path))

        # get point cloud
        # NOTE: skip convert process if point cloud files already exist
        if config.SKIP_CONVERT and os.path.exists(complete_pcd_path):

            # read point cloud file
            complete_pcd   = util.read_pcd(complete_pcd_path)

        else:

            # read mesh file
            complete_mesh   = util.read_mesh(complete_mesh_path)

            # convert mesh to point cloud by sampling
            complete_pcd    = util.convert_mesh2pcd(complete_mesh)

            # write auto-triangulate mesh file
            util.write_mesh(triangle_mesh_path, complete_mesh)

            # write point cloud file
            util.write_pcd(complete_pcd_path  , complete_pcd)

        # loop over virtual scans
        for vscan_index in range(config.VSCAN_NUM):

            # read incomplete(virtual scan) point cloud file
            incomplete_pcd_path = os.path.join(pcd_path , config.PCD_FILENAME_VSCAN.format(vscan_index))
            incomplete_pcd = util.read_pcd_vscan(incomplete_pcd_path)

            # visualize point clouds
            if not config.SKIP_VISUALIZE:
                util.visualize_pcd(complete_pcd)
                util.visualize_pcd(incomplete_pcd)

            # convert data
            complete_data   = np.asarray(complete_pcd.points  , dtype=np.float32)
            incomplete_data = np.asarray(incomplete_pcd.points, dtype=np.float32)
            label_data      = np.asarray(class_index          , dtype=np.int32)

            # augment data
            if config.AUGMENT_DATA:
                for _ in range(config.AUGMENT_NUMBER):
                    complete_data_, incomplete_data_ = util.augment_data(complete_data, incomplete_data)
                    complete_pcds  .append(complete_data_)
                    incomplete_pcds.append(incomplete_data_)
                    labels         .append(label_data)
            else:
                complete_pcds  .append(complete_data)
                incomplete_pcds.append(incomplete_data)
                labels         .append(label_data)

# stack data
complete_pcds   = np.stack(complete_pcds)
incomplete_pcds = np.stack(incomplete_pcds)
labels          = np.stack(labels)

"""
Create CRN Dataset
"""

# set data size
data_size = complete_pcds.shape[0]
train_data_size = data_size * config.DATASET_RATE_TRAIN // 100
valid_data_size = data_size * config.DATASET_RATE_VALID // 100
test_data_size  = data_size * config.DATASET_RATE_TEST  // 100

# set data indices
train_data_index_start = 0
train_data_index_end   = train_data_index_start + train_data_size
valid_data_index_start = train_data_index_end
valid_data_index_end   = valid_data_index_start + valid_data_size
test_data_index_start  = valid_data_index_end
test_data_index_end    = data_size

# create data directory
if not os.path.exists(config.HDF5_DIRECTORY):
    os.mkdir(config.HDF5_DIRECTORY)

# write data file (HDF5 format)
util.write_data(os.path.join(config.HDF5_DIRECTORY, config.HDF5_FILENAME_TRAIN),
                complete_pcds  [train_data_index_start:train_data_index_end, :, :],
                incomplete_pcds[train_data_index_start:train_data_index_end, :, :],
                labels         [train_data_index_start:train_data_index_end])
util.write_data(os.path.join(config.HDF5_DIRECTORY, config.HDF5_FILENAME_VALID),
                complete_pcds  [valid_data_index_start:valid_data_index_end, :, :],
                incomplete_pcds[valid_data_index_start:valid_data_index_end, :, :],
                labels         [valid_data_index_start:valid_data_index_end])
util.write_data(os.path.join(config.HDF5_DIRECTORY, config.HDF5_FILENAME_TEST),
                complete_pcds  [test_data_index_start:test_data_index_end, :, :],
                incomplete_pcds[test_data_index_start:test_data_index_end, :, :],
                labels         [test_data_index_start:test_data_index_end])
