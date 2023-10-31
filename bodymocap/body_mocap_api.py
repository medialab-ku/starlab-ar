# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import sys
import torch
import numpy as np
import pickle 
from torchvision.transforms import Normalize
import time 

from bodymocap.models.head.smplx_cam_head import SMPLXCamHead
from bodymocap.models import hmr, SMPL, SMPLX, HMR, OneEuroFilter
from bodymocap import constants
from bodymocap.utils.train_utils import load_pretrained_model
from bodymocap.utils.imutils import crop, crop_bboxInfo, process_image_bbox, process_image_keypoints, bbox_from_keypoints
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
import mocap_utils.geometry_utils as gu

from scipy.spatial.transform import Rotation



class BodyMocap(object):
    def __init__(self, regressor_checkpoint, smpl_dir, device=torch.device('cuda'), use_smplx=False, posebert = None, posebert_len = 64):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load parametric model (SMPLX or SMPL)
        if use_smplx:
            smplModelPath = smpl_dir + '/SMPLX_NEUTRAL.pkl'
            self.smpl = SMPLX(smpl_dir,
                    batch_size=1,
                    num_betas = 10,
                    use_pca = False,
                    create_transl=False).to(self.device)
            self.use_smplx = True
        else:
            smplModelPath = smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
            self.smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to(self.device)
            self.use_smplx = False
            self.smplx_cam_head = SMPLXCamHead(img_res=224).to(device)

        #Load pre-trained neural network 
        SMPL_MEAN_PARAMS = './extra_data/body_module/data_from_spin/smpl_mean_params.npz'
        # self.model_regressor = hmr(SMPL_MEAN_PARAMS).to(self.device)
        # self.model_regressor = HMR().to(self.device)   #mocapSPIN
        # checkpoint = torch.load(regressor_checkpoint)
        # self.model_regressor.load_state_dict(checkpoint['model'], strict=False)

        self.left_hand_pose = torch.eye(3, device=device, dtype=torch.float32).view(
                1, 1, 3, 3)
        
        #hmr hrnet48
        # self.model_regressor = HMR(
        #     backbone="hrnet_w48-conv",
        #     img_res=224,
        #     pretrained_ckpt="scratch",
        #     useCLIFF = True,
        # ).to(self.device)

        self.model_regressor = HMR(
            backbone="convnext",
            img_res=224,
            pretrained_ckpt="scratch",
            useCLIFF = False,
        ).to(self.device)
        cpkt = torch.load(regressor_checkpoint)['state_dict']
        load_pretrained_model(self.model_regressor, cpkt, overwrite_shape_mismatch=True, remove_lightning=True)

        self.model_regressor.eval()
        self.posebert = posebert
        self.pb_len = posebert_len
        self.queue_pose = []
        self.shape = None
        self.cam = None
        self.count = 0
        self.boxScale_o2n = None
        self.bboxTopLeft = None

        for i in range(self.pb_len):
            self.queue_pose.append(torch.zeros((1, 24, 3, 3)).float())
        
        self.filter_config_3d = {
        'freq': 30,        # system frequency about 30 Hz
        'mincutoff': 1,  # value refer to the paper
        'beta': 0.4,       # value refer to the paper
        'dcutoff': 0.4     # not mentioned, empirically set
        }
        self.filter_3d = (OneEuroFilter(**self.filter_config_3d),
                OneEuroFilter(**self.filter_config_3d),
                OneEuroFilter(**self.filter_config_3d))

        self.detection = (OneEuroFilter(**self.filter_config_3d),
                OneEuroFilter(**self.filter_config_3d),
                OneEuroFilter(**self.filter_config_3d),
                OneEuroFilter(**self.filter_config_3d))
        
    def regress(self, img_original, body_bbox_list):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                body_bbox: bounding box around the target: (minX, minY, width, height)
            outputs:
                pred_vertices_img:
                pred_joints_vis_img:
                pred_rotmat
                pred_betas
                pred_camera
                bbox: [bbr[0], bbr[1],bbr[0]+bbr[2], bbr[1]+bbr[3]])
                bboxTopLeft:  bbox top left (redundant)
                boxScale_o2n: bbox scaling factor (redundant) 
        """
        pred_output_list = list()
        self.count +=1
        bbox_scale = []
        bbox_center = []

        for body_bbox in body_bbox_list:
            t = time.time()
            body_bbox[0] = self.detection[0](body_bbox[0], t)
            body_bbox[1] = self.detection[1](body_bbox[1], t)
            body_bbox[2] = self.detection[2](body_bbox[2], t)
            body_bbox[3] = self.detection[3](body_bbox[3], t)
            img, norm_img, boxScale_o2n, bboxTopLeft, bbox = process_image_bbox(
                img_original, body_bbox, input_res=constants.IMG_RES)
            bboxTopLeft = np.array(bboxTopLeft)
            bbox_scale.append(bbox['scale'])
            bbox_center.append([bbox["center"][0], bbox["center"][1]])
            # bboxTopLeft = bbox['bboxXYWH'][:2]
            if img is None:
                pred_output_list.append(None)
                continue

            orig_height, orig_width = img_original.shape[:2]
            bbox_center = torch.tensor(bbox_center).cuda().float()
            bbox_scale = torch.tensor(bbox_scale).cuda().float()
            img_h = torch.tensor(orig_height).repeat(norm_img.shape[0]).cuda().float()
            img_w = torch.tensor(orig_width).repeat(norm_img.shape[0]).cuda().float()

            with torch.no_grad():
                # model forward
                # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
                hmr_output = self.model_regressor(norm_img.to(self.device), bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
                pred_rotmat, pred_betas, pred_camera = hmr_output['pred_pose'], hmr_output['pred_shape'], hmr_output['pred_cam']
                pred_rotmat = torch.cat((pred_rotmat, self.left_hand_pose.repeat(1, 2, 1, 1)), dim=1)
                if self.count == 20:
                    # self.cam = hmr_output['pred_cam']
                    # # self.cam = pred_camera
                    # self.bboxTopLeft = bboxTopLeft
                    # self.boxScale_o2n = boxScale_o2n
                    self.shape = pred_betas
                if self.posebert is not None:
                    self.queue_pose.pop(0)
                    self.queue_pose.append(pred_rotmat.float().cpu())
                    rot_seq = torch.cat([self.queue_pose[j] for j in range(len(self.queue_pose)-1)]).to(self.device)
                    rot_seq = self.posebert(rotmat=rot_seq.unsqueeze(0))
                    pred_rotmat = rot_seq[0][-1].unsqueeze(0)
                if self.cam is not None:
                    # hmr_output['pred_cam'] = self.cam
                    # # pred_camera = self.cam
                    # bboxTopLeft = self.bboxTopLeft
                    # boxScale_o2n = self.boxScale_o2n
                    pred_betas = self.shape


                hmr_output['pred_pose'] = pred_rotmat[:,:22,:,:]
                hmr_output['pred_shape'] = pred_betas
                smpl_output = self.model_regressor.smpl(
                rotmat=hmr_output['pred_pose'],
                shape=hmr_output['pred_shape'])
                smpl_output.update(hmr_output)

                # Convert rot_mat to aa since hands are always in aa
                # pred_aa = rotmat3x3_to_angle_axis(pred_rotmat)
                # pred_rotmat = torch.cat((pred_rotmat, self.random_rotations_cuda.repeat(1, 2, 1, 1)), dim=1)
                # pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
                # pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
 
                # smpl_output = self.smpl(
                #     betas=pred_betas, 
                #     body_pose=pred_aa[:,3:],
                #     global_orient=pred_aa[:,:3], 
                #     pose2rot=True)

                # pred_vertices = smpl_output.vertices
                pred_vertices = smpl_output['vertices']
                # pred_joints_3d = smpl_output.joints

                pred_vertices = pred_vertices[0].cpu().numpy()
                # t = time.time()
                # hmr_output['pred_cam'][0, 0] = self.filter_3d[0](hmr_output['pred_cam'][0, 0], t)
                # hmr_output['pred_cam'][0, 1] = self.filter_3d[1](hmr_output['pred_cam'][0, 1], t)
                # hmr_output['pred_cam'][0, 2] = self.filter_3d[2](hmr_output['pred_cam'][0, 2], t)

                pred_camera = hmr_output['pred_cam'].cpu().numpy().ravel()
                # pred_camera = pred_camera.cpu().numpy().ravel()
                camScale = pred_camera[0] # *1.15
                camTrans = pred_camera[1:]

                pred_output = dict()
                # # Convert mesh to original image space (X,Y are aligned to image)
                # # 1. SMPL -> 2D bbox
                # # 2. 2D bbox -> original 2D image
                pred_vertices_bbox = convert_smpl_to_bbox(pred_vertices, camScale, camTrans)
                pred_vertices_img = convert_bbox_to_oriIm(
                    pred_vertices_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])

                # # Convert joint to original image space (X,Y are aligned to image)
                # pred_joints_3d = pred_joints_3d[0].cpu().numpy() # (1,49,3)
                # pred_joints_vis = pred_joints_3d[:,:3]  # (49,3)
                # pred_joints_vis_bbox = convert_smpl_to_bbox(pred_joints_vis, camScale, camTrans) 
                # pred_joints_vis_img = convert_bbox_to_oriIm(
                #     pred_joints_vis_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0]) 

                # # Output
                pred_output['img_cropped'] = img[:, :, ::-1]
                pred_output['pred_vertices_img'] = pred_vertices_img # SMPL vertex in image space
                # pred_output['pred_joints_img'] = pred_joints_vis_img # SMPL joints in image space
                # pred_output['faces'] = self.smpl.faces
                pred_output['faces'] = self.smplx_cam_head.smplx.faces
            
                # if self.use_smplx:
                #     img_center = np.array((img_original.shape[1], img_original.shape[0]) ) * 0.5
                #     # right hand
                #     pred_joints = smpl_output.right_hand_joints[0].cpu().numpy()     
                #     pred_joints_bbox = convert_smpl_to_bbox(pred_joints, camScale, camTrans)
                #     pred_joints_img = convert_bbox_to_oriIm(
                #         pred_joints_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])
                #     pred_output['right_hand_joints_img_coord'] = pred_joints_img
                #     # left hand 
                #     pred_joints = smpl_output.left_hand_joints[0].cpu().numpy()
                #     pred_joints_bbox = convert_smpl_to_bbox(pred_joints, camScale, camTrans)
                #     pred_joints_img = convert_bbox_to_oriIm(
                #         pred_joints_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])
                #     pred_output['left_hand_joints_img_coord'] = pred_joints_img
                
                pred_output_list.append(pred_output)

        return pred_output_list, img_h, img_w
    

    def get_hand_bboxes(self, pred_body_list, img_shape):
        """
            args: 
                pred_body_list: output of body regresion
                img_shape: img_height, img_width
            outputs:
                hand_bbox_list: 
        """
        hand_bbox_list = list()
        for pred_body in pred_body_list:
            hand_bbox = dict(
                left_hand = None,
                right_hand = None
            )
            if pred_body is None:
                hand_bbox_list.append(hand_bbox)
            else:
                for hand_type in hand_bbox:
                    key = f'{hand_type}_joints_img_coord'
                    pred_joints_vis_img = pred_body[key]

                    if pred_joints_vis_img is not None:
                        # get initial bbox
                        x0, x1 = np.min(pred_joints_vis_img[:, 0]), np.max(pred_joints_vis_img[:, 0])
                        y0, y1 = np.min(pred_joints_vis_img[:, 1]), np.max(pred_joints_vis_img[:, 1])
                        width, height = x1-x0, y1-y0
                        # extend the obtained bbox
                        margin = int(max(height, width) * 0.2)
                        img_height, img_width = img_shape
                        x0 = max(x0 - margin, 0)
                        y0 = max(y0 - margin, 0)
                        x1 = min(x1 + margin, img_width)
                        y1 = min(y1 + margin, img_height)
                        # result bbox in (x0, y0, w, h) format
                        hand_bbox[hand_type] = np.array([x0, y0, x1-x0, y1-y0]) # in (x, y, w, h ) format

                hand_bbox_list.append(hand_bbox)

        return hand_bbox_list
