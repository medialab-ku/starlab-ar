#pragma once

#define KF_DEBUG				false

#define KF_MAP_BLOCK_DIM_X		32
#define KF_MAP_BLOCK_DIM_Y		32

#define KF_VOL_BLOCK_DIM_X		16
#define KF_VOL_BLOCK_DIM_Y		16
#define KF_VOL_BLOCK_DIM_Z		4

#define KF_BF_WINDOW_SIZE		7
#define KF_BF_SIGMA_DEPTH		0.07f
#define KF_BF_SIGMA_SPACE		5.0f

#define KF_SS_DOWN_SCALE		2
#define KF_SS_SIGMA_DEPTH		0.2f

#define KF_FUSE_SDF_RANGE		3
#define KF_FUSE_WEIGHT_SCALE	1.0f
#define KF_FUSE_WEIGHT_THRESH	0.3f
#define KF_FUSE_WEIGHT_MAX		200.0f

#define KF_RC_NEAR_Z			0.5f
#define KF_RC_FAR_Z				4.5f
#define KF_RC_STEP_SLOW			1.0f
#define KF_RC_STEP_FAST			4.0f

#define KF_ICP_EPS_DIST			0.05f
#define KF_ICP_EPS_ANGLE		0.7f
#define KF_ICP_ITER_NUM			{ 10, 5, 4 }
