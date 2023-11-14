import torch
import torch.nn as nn

from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48
from .backbone.resnet import resnet50
from .backbone.convnext import convnext_tiny

from .head.hmr_head_orig import HMRHeadOrig
from .head.hmr_head_cliff import HMRHeadCLIFF
from .head.smplx_cam_head import SMPLXCamHead
from .head.smplx_cam_head_proj import SMPLXCamHeadProj
from .head.smplx_head import SMPLXHead
from ..core.config import PRETRAINED_CKPT_FOLDER

extra_data_folder = '/home/svvj/workspace/frankmocap/extra_data/'
PRETRAINED_CKPT_FOLDER = {
    'hrnet_w32-coco': extra_data_folder + 'body_module/pretrained_weights/hrnet/pose_hrnet_w32_256x192.pth',
    'hrnet_w32-imagenet': extra_data_folder + 'body_module/pretrained_weights/hrnet/hrnetv2_w32_imagenet_pretrained.pth',
    'hrnet_w32-scratch': '',
    'hrnet_w48-coco': extra_data_folder + 'body_module/pretrained_weights/hrnet/pose_hrnet_w48_256x192.pth',
    'hrnet_w48-imagenet': extra_data_folder + 'body_module/pretrained_weights/hrnet/hrnetv2_w48_imagenet_pretrained.pth',
    'hrnet_w48-scratch': '',
    'resnet50-coco': 'data/ckpt/pretrained/pose_resnet50_256x192.pth',
    'resnet50-scratch': '',
    'convnext-coco': 'data/ckpt/pretrained/convnext_tiny_22k_224.pth',
    'convnext-scratch': '',
}

class HMR(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            img_res=224,
            focal_length=5000,
            pretrained_ckpt=None,
            useCLIFF=None
    ):
        super(HMR, self).__init__()
        self.useCLIFF = useCLIFF


        # Initialize backbone
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            pretrained_ckpt = backbone + '-' + pretrained_ckpt
            pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
            self.backbone = eval(backbone)(
                pretrained_ckpt_path=pretrained_ckpt_path,
                downsample=True,
                use_conv=(use_conv == 'conv'),
            ) 
        # Todo support resnet
        elif backbone.startswith('convnext'):
            backbone = backbone
            pretrained_ckpt = backbone + '-' + pretrained_ckpt
            pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
            self.backbone = convnext_tiny(
                pretrained_ckpt_path=pretrained_ckpt_path,
                pretrained=False
            )    
        else:
            backbone = backbone
            pretrained_ckpt = backbone + '-' + pretrained_ckpt
            pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
            self.backbone = eval(backbone)(
                pretrained_ckpt_path=pretrained_ckpt_path,
                pretrained=False, 
                progress=False
            ) 
        # Run on real images used in original CLIFF
        # if hparams.TRIAL.version == 'real':
        #     if hparams.TRIAL.bedlam_bbox:
        #         self.head = HMRHeadCLIFFSMPL(
        #             num_input_features=get_backbone_info(backbone)['n_output_channels'],
        #             backbone=backbone,
        #         )
        #         self.smpl = SMPLCamHead(img_res=img_res)

        # else:
        if self.useCLIFF:
            self.head = HMRHeadCLIFF(
                num_input_features=get_backbone_info(backbone)['n_output_channels'],
                backbone=backbone,
            )
            if False:
                self.smpl = SMPLXCamHeadProj(img_res=img_res) 
            else:
                self.smpl = SMPLXCamHead(img_res=img_res)

        else:
            self.head = HMRHeadOrig(
                num_input_features=get_backbone_info(backbone)['n_output_channels'],
                backbone=backbone,
            )
            self.smpl = SMPLXHead(
                focal_length=focal_length,
                img_res=img_res)

    def forward(
            self,
            images,
            bbox_scale=None,
            bbox_center=None,
            img_w=None,
            img_h=None,
            fl=None
    ):
        batch_size = images.shape[0]

        if fl is not None:
            # GT focal length
            focal_length = fl
        else:
            # Estimate focal length
            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            focal_length = focal_length.repeat(2).view(batch_size, 2)

        # Initialze cam intrinsic matrix
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0] = focal_length[:, 0]
        cam_intrinsics[:, 1, 1] = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w/2.
        cam_intrinsics[:, 1, 2] = img_h/2.

        if self.useCLIFF:
            # Taken from CLIFF repository
            cx, cy = bbox_center[:, 0], bbox_center[:, 1]
            b = bbox_scale * 200 / 1.0
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                    dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)   # [-1, 1]
            bbox_info[:, 2] = bbox_info[:, 2] / cam_intrinsics[:, 0, 0]  # [-1, 1]
            bbox_info = bbox_info.cuda().float()
            features = self.backbone(images)
            hmr_output = self.head(features, bbox_info=bbox_info)
        else:
            features = self.backbone(images)
            hmr_output = self.head(features)

        # if self.useCLIFF:
        #     # Assuming prediction are in camera coordinate
        #     smpl_output = self.smpl(
        #         rotmat=hmr_output['pred_pose'],
        #         shape=hmr_output['pred_shape'],
        #         # cam=hmr_output['pred_cam'],
        #         # cam_intrinsics=cam_intrinsics,
        #         # bbox_scale=bbox_scale,
        #         # bbox_center=bbox_center,
        #         # img_w=img_w,
        #         # img_h=img_h,
        #         # normalize_joints2d=False,
        #     )
        # else:
        #     smpl_output = self.smpl(
        #         rotmat=hmr_output['pred_pose'],
        #         shape=hmr_output['pred_shape'],
        #         # cam=hmr_output['pred_cam'],
        #         # normalize_joints2d=True,
        #     )
        # smpl_output.update(hmr_output)
        return hmr_output
