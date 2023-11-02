# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import taichi as ti
import meshtaichi_patcher as patcher
import numpy as np
import cv2
import argparse
import json
import pickle
import cyobj.io as mio
from datetime import datetime

from demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
from bodymocap.posebert import PoseBERT
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer

from physics.mesh import Mesh, applyTransform, makeBox
from physics.solver import Solver
from physics.math import create_batch_eyes

import renderer.image_utils as imu
from renderer.viewer2D import ImShow

ti.init(kernel_profiler=True, arch=ti.cuda, device_memory_GB=16)

window = ti.ui.Window("Display Mesh", (640, 480), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.0, 1.0, 5.0)
camera.fov(30)
camera.up(0, 1, 0)

save_video = False
result_dir = "./taichi_output"
video_manager = ti.tools.VideoManager(output_dir=result_dir+'/video', framerate=30, automatic_build=False)

dt = 0.003
mesh = Mesh("obj_files/square_big.obj", scale=0.1, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
static_mesh = Mesh("obj_files/dummy_human.obj", scale=1.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))

total_min_max = ti.Vector.field(3, dtype=ti.f32, shape=2)
box_v = ti.Vector.field(3, dtype=ti.f32, shape=8)
box_i = ti.Vector.field(2, dtype=ti.i32, shape=12)
box_i_np = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7],
                     [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
box_v.fill(0.0)
box_i.from_numpy(box_i_np)

gbox_v = ti.Vector.field(3, dtype=ti.f32, shape=8)
g_min_max = ti.Vector.field(3, dtype=ti.f32, shape=2)
g_min_max_np = np.array([[-2.0, -0.5, -2.0], [2.0, 2.0, 2.0]])
g_min_max.from_numpy(g_min_max_np)

sim = Solver(mesh, static_mesh=static_mesh, min_range=g_min_max[0], max_range=g_min_max[1], dt=dt, max_iter=1)
makeBox(g_min_max, gbox_v)

human_verts_ti = ti.Vector.field(3, dtype=ti.f32, shape=10475)
human_faces_ti = ti.field(dtype=ti.i32, shape=(20908 * 3,))

def run_body_mocap(args, body_bbox_detector, body_mocap, visualizer=None):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)
    cur_frame = args.start_frame
    video_frame = 0
    sim_frame = 0
    
    run_sim = True
    
    timer = Timer()
    while True:
        timer.tic()
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':    
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        is_first_frame = False
        if cur_frame is args.start_frame:
            is_first_frame = True

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")
        print("Frame: ", cur_frame)

        if load_bbox:
            body_pose_list = None
        else:
            body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
                img_original_bgr)
        hand_bbox_list = [None, ] * len(body_bbox_list)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue

        #Sort the bbox using bbox size 
        # (to make the order as consistent as possible without tracking)
        bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
        if args.single_person and len(body_bbox_list)>0:
            body_bbox_list = [body_bbox_list[0], ]       

        # Body Pose Regression
        verts_torch, faces_np, img_h, img_w = body_mocap.regress(img_original_bgr, body_bbox_list)
        '''
        # assert len(body_bbox_list) == len(pred_output_list)
        # cv2.imwrite(args.out_dir + '/test' + str(video_frame) + '.png' ,pred_output_list[0]['img_cropped'])

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
        '''
        
        human_verts_ti.from_torch(verts_torch)
        min_torch = torch.min(verts_torch, dim=0).values.reshape(1, 3)
        max_torch = torch.max(verts_torch, dim=0).values.reshape(1, 3)
        total_min_max_torch = torch.cat([min_torch, max_torch], dim=0)
        total_min_max.from_torch(total_min_max_torch)
        applyTransform(total_min_max, scale=1.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(180.0, 0.0, 0.0))
        makeBox(total_min_max, box_v)

        applyTransform(human_verts_ti, scale=1.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(180.0, 0.0, 0.0))   # rotate 180 degrees
        if is_first_frame:
            human_faces_ti.from_numpy(faces_np.reshape(-1))

        # Export mesh
        # if video_frame >= 100:
        #     mesh_frame = video_frame - 100
        #     filepath = "seq_files/body_seq/dummy_" + str(mesh_frame).zfill(4) + ".obj"
        #     if not os.path.exists(filepath):
        #         open(filepath, 'w').close()
        #     mio.write_obj(filepath, human_verts_ti.to_numpy().astype(np.float64), human_faces_ti.to_numpy().reshape(-1, 3).astype(np.int_))

        sim.update_static_mesh(human_verts_ti)

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                run_sim = not run_sim

            if window.event.key == 'r':
                sim.reset()
                run_sim = False

        if run_sim:
            sim_frame += 1
            sim.update(dt=dt, num_sub_steps=20)

        # Render
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        camera.lookat(0.0, 0.5, 0.0)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(-0.5, 3.0, 3.0), color=(0.3, 0.3, 0.3))
        scene.point_light(pos=(0.5, 3.0, 3.0), color=(0.3, 0.3, 0.3))
        # scene.particles(human_verts_ti, radius=0.01, color=(0.5, 0.5, 0.5))
        scene.mesh(vertices=human_verts_ti, indices=human_faces_ti, color=(0.5, 0.5, 0.5))
        scene.mesh(vertices=sim.verts.x, indices=sim.face_indices, color=(0.3, 0.5, 0.2))
        scene.lines(gbox_v, width=1.0, indices=box_i, color=(0.0, 1.0, 0.0))
        scene.lines(box_v, width=1.0, indices=box_i, color=(1.0, 0.0, 0.0))
        canvas.scene(scene)

        if save_video:
            img = window.get_image_buffer_as_numpy()
            video_manager.write_frame(img)
        
        window.show()

        '''
        # visualization
        if visualizer is not None:
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list = pred_mesh_list, 
                body_bbox_list = body_bbox_list)

        
            # show result in the screen
            if not args.no_display:
                res_img = res_img.astype(np.uint8)
                ImShow(res_img)

            # save result image
            # if args.out_dir is not None and args.save_frame:
            #     demo_utils.save_res_img(args.out_dir, image_path, res_img)

            # save predictions to pkl

        # if args.save_pred_pkl:
        #     demo_type = 'body'
        #     demo_utils.save_pred_to_pkl(
        #         args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)
        '''

        timer.toc(bPrint=True,title="Time")
        print(f"Processed : {image_path}")

    #save images as a video
    # if not args.no_video_out and input_type in ['video', 'webcam']:
    #     demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

    video_manager.make_video(gif=True, mp4=True)


def main():
    args = DemoOptions().parse()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator()

    # Set Posebert 
    ckpt_posebert = torch.load(args.ckpt_posebert_fn, map_location=device)
    SMPL_MEAN_PARAMS = './extra_data/body_module/data_from_spin/smpl_mean_params.npz'
    init_pose = torch.from_numpy(np.load(SMPL_MEAN_PARAMS)['pose'][:]).unsqueeze(0)
    posebert = PoseBERT(init_pose=init_pose, in_dim=24*6)
    posebert = posebert.eval()
    poserbert = posebert.to(device)
    posebert_seq_len = 64
    posebert.load_state_dict(ckpt_posebert['model_state_dict'])
    # poserbert = None

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx, poserbert, posebert_seq_len)


    # Set Visualizer
    print('Renderer type:', args.renderer_type)
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    elif args.renderer_type == 'taichi':
        from renderer.taichi_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)


    run_body_mocap(args, body_bbox_detector, body_mocap, visualizer)


if __name__ == '__main__':
    main()