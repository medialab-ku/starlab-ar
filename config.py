"""
Config
"""

SKIP_CONVERT   = True
SKIP_VISUALIZE = True

DATASET_PATH = "ShapeNetCore.v2"

MESH_DIRECTORY = "models"
MESH_FILENAME_ORG  = "model_normalized.obj"
MESH_FILENAME_TRI  = "model_normalized_tri.obj"

PCD_DIRECTORY           = "pcds"
PCD_FILENAME_COMPLETE   = "pcd_complete.ply"
PCD_FILENAME_INCOMPLETE = "pcd_incomplete.ply"
PCD_FILENAME_VSCAN      = "pcd_vscan_{}.txt"

CLASS_IDS = [""] * 8 + ["02946921"]

VSCAN_NUM = 20

SAMPLING_UNIFORM = 2048 * 5
SAMPLING_POISSON = 2048

AUGMENT_DATA        = False
AUGMENT_NUMBER      = 20
AUGMENT_TRANSLATION = [-0.2, 0.2]
# AUGMENT_ROTATION    = []
AUGMENT_SCALING     = [0.8, 1.2]

DATASET_RATE_TRAIN = 80
DATASET_RATE_VALID = 10
DATASET_RATE_TEST  = 10

HDF5_DIRECTORY                    = "data"
HDF5_FILENAME_TRAIN               = "train_data.h5"
HDF5_FILENAME_VALID               = "valid_data.h5"
HDF5_FILENAME_TEST                = "test_data.h5"
HDF5_DATASET_NAME_COMPLETE_PCDS   = "complete_pcds"
HDF5_DATASET_NAME_INCOMPLETE_PCDS = "incomplete_pcds"
HDF5_DATASET_NAME_LABELS          = "labels"
