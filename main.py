import os


############################## Directory Setting ##############################


DIR_IMAGE = 'image'
DIR_VIDEO = 'video'
DIR_MESH = 'mesh'

for directory in [DIR_IMAGE, DIR_VIDEO, DIR_MESH]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


############################## Parameter Setting ##############################


# TUM RGB-D
# DATA_NAME = 'rgbd_dataset_freiburg1_xyz'
# MAP_SIZE = (640, 480)
# CAM_INTRINSIC = (591.1, 590.1, 331.0, 234.0)
# PYRAMID_LEVEL = 3

# RealSense D455 (color to depth align)
# DATA_NAME = 'rs2'
# MAP_SIZE = (1280, 720)
# CAM_INTRINSIC = (642.2863159179688, 642.2863159179688, 635.0969848632812, 360.14208984375)
# PYRAMID_LEVEL = 3

# RealSense D455 (depth to color align)
DATA_NAME = 'rs2'
MAP_SIZE = (1280, 800)
CAM_INTRINSIC = (637.4338989257812, 636.5624389648438, 637.0335693359375, 410.9555358886719)
PYRAMID_LEVEL = 3

BG_VOLUME_SIZE = (128, 128, 128)
BG_CENTER_POS = (0.0, 0.0, 2.0)
BG_VOXEL_SIZE = 0.02

OBJ_VOLUME_SIZE = 32
OBJ_VOLUME_PERCENT = 10
OBJ_VOLUME_SCALE = 1.5

CHECK_SCORE = 0.5
CHECK_IOU_MAX = 0.5
CHECK_CENTER = 0.05
CHECK_SIZE = 0.1


############################## Main Process ##############################


# create data loader
data_loader = data.tum.DataLoader(DATA_NAME)

# create frame
frame = slam.kf.Frame(MAP_SIZE, PYRAMID_LEVEL, CAM_INTRINSIC)
background = slam.kf.Object(BG_VOLUME_SIZE, BG_CENTER_POS, BG_VOXEL_SIZE)
camera = slam.kf.Camera(MAP_SIZE, CAM_INTRINSIC)

# create object dictionary
object_dict = {}

# run segmentation fusion
while not data_loader.empty():

    # get frame data
    _, rgb, _, depth = data_loader.next()

    # show RGB and Depth images
    cv2.imshow('Color', rgb)
    cv2.imshow('Depth', depth / 4)

    # run RGB segmentation
    segmentations = seg.mrcnn.run(rgb)

    # get segmentation results
    num_instances = len(segmentations)
    scores = segmentations.scores
    pred_masks = segmentations.pred_masks
    pred_classes = segmentations.pred_classes

    # visualize segmentation results
    segment_viz = seg.mrcnn.visualize(rgb, segmentations)
    cv2.imshow('Segmentation', segment_viz)

    # run depth preprocess
    frame.preprocess(depth)
    vertex_map = frame.get_vertex_map()

    # run background fusion
    frame.clear()
    background.integrate(frame)

    # run camera tracking
    camera.clear()
    camera.raycast(background)
    # camera.track(frame)  # FIXME: turn off tracking

    # get instance map
    camera.clear()
    for object in object_dict.values():
        camera.raycast(object)
    instance_map = camera.get_instance_map()

    # visualize instance map
    cv2.imshow('Instance Map', instance_map / 5.0)

    # run object fusion
    for score, mask, cls in zip(scores, pred_masks, pred_classes):

        # check score
        if score < CHECK_SCORE:
            continue

        # convert mask (from Tensor to NumPy)
        mask = mask.numpy()

        # erode mask (remove mask edge noise)
        mask = binary_erosion(mask, iterations=4)

        # compute IoU for each object
        iou_dict = {}
        for id, object in object_dict.items():

            instance_mask = (instance_map == id)

            inter = instance_mask * mask
            union = instance_mask + mask

            iou = inter.sum() / union.sum()
            iou_dict[id] = iou

        # get max IoU
        if len(iou_dict) > 0:
            iou_max_key = max(iou_dict, key=iou_dict.get)
            iou_max_value = iou_dict[iou_max_key]
        else:
            iou_max_key = 0
            iou_max_value = 0.0

        # check max IoU
        if iou_max_value > CHECK_IOU_MAX:

            # get object in list
            object = object_dict[iou_max_key]

        else:

            # get vertex in mask
            vertex_mask = vertex_map[mask]
            vertex_mask = vertex_mask[~np.all(vertex_mask == 0., axis=1)]

            # check vertex exist
            if vertex_mask.size == 0:
                continue

            # compute volume center and size
            vertex_mask_min = np.percentile(vertex_mask, OBJ_VOLUME_PERCENT, axis=0)
            vertex_mask_max = np.percentile(vertex_mask, 100 - OBJ_VOLUME_PERCENT, axis=0)
            center = (vertex_mask_max + vertex_mask_min) / 2
            size = np.max(vertex_mask_max - vertex_mask_min) * OBJ_VOLUME_SCALE

            # TSDF culling process
            culling = False
            for object in object_dict.values():
                center_diff = np.linalg.norm(np.array(object.center) - np.array(center))
                size_diff = abs(object.voxel_size * OBJ_VOLUME_SIZE - size)
                if center_diff < CHECK_CENTER and size_diff < CHECK_SIZE:
                    culling = True
                    break
            if culling:
                continue

            # create new object
            object = slam.kf.Object(
                (OBJ_VOLUME_SIZE, OBJ_VOLUME_SIZE, OBJ_VOLUME_SIZE),
                (center[0], center[1], center[2]),
                size / OBJ_VOLUME_SIZE)
            object_dict[object.id] = object

        # apply object fusion
        frame.mask(mask)
        object.integrate(frame)

    # update visualization
    cv2.waitKey(1)

# close all windows
cv2.destroyAllWindows()

# write background mesh
# util.write_tsdf_mesh(os.path.join(DIR_MESH, "mesh1_tsdf.obj"), background.get_tsdf_volume())
object_dict[1] = background

# convert all objects to meshes
mesh_dict = {}
for id, object in object_dict.items():

    # get object vertices
    vertices = object.get_polygon_volume()
    vertices = np.reshape(vertices, (-1, 3))
    vertices = vertices[~np.all(vertices == 0, axis=1)]
    vertices[:, 1:3] = -vertices[:, 1:3] # NOTE: yz-negation

    # create object triangles
    triangles = np.reshape(np.arange(vertices.shape[0]), (-1, 3))

    # create object mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # add mesh to dictionary
    mesh_dict[id] = mesh

    # write object mesh
    util.write_mesh(os.path.join(DIR_MESH, f"mesh{id}_poly.obj"), mesh)
    util.write_tsdf_mesh(os.path.join(DIR_MESH, f"mesh{id}_tsdf.obj"), object.get_tsdf_volume())

# apply completion for some objects
shinv = {
    'can': cmpl.shinv.ShapeInversion('can'),
    'ball': cmpl.shinv.ShapeInversion('ball'),
}
for id, mesh in mesh_dict.items():

    # check id and give its class name
    class_name = None
    if id == 2:
        class_name = 'ball'
    if id == 4:
        class_name = 'can'

    # check valid class name
    if not class_name:
        continue

    # apply completion process (ShapeInversion)
    pcd_orig = util.convert_mesh2pcd(mesh)
    pts_orig = util.convert_pcd2pts(pcd_orig)
    pts_orig_norm, mean, norm = util.normalize_pts(pts_orig)
    pts_cmpl_norm = shinv[class_name].run(pts_orig_norm)
    pts_cmpl = pts_cmpl_norm * norm + mean
    pcd_cmpl = util.convert_pts2pcd(pts_cmpl)

    # show and write point cloud
    util.show_pcd(pcd_orig)
    util.show_pcd(pcd_cmpl)
    util.write_pcd(os.path.join(DIR_MESH, f"pcd{id}_orig.ply"), pcd_orig)
    util.write_pcd(os.path.join(DIR_MESH, f"pcd{id}_cmpl.ply"), pcd_cmpl)

    # convert point cloud to mesh
    mesh_orig = util.convert_pcd2mesh(pcd_orig, norm)
    mesh_cmpl = util.convert_pcd2mesh(pcd_cmpl, norm)

    # show and write mesh file
    util.show_mesh(mesh_orig)
    util.show_mesh(mesh_cmpl)
    util.write_mesh(os.path.join(DIR_MESH, f"mesh{id}_orig.obj"), mesh_orig)
    util.write_mesh(os.path.join(DIR_MESH, f"mesh{id}_cmpl.obj"), mesh_cmpl)
