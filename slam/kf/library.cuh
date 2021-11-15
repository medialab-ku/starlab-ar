#pragma once

#include "kf.cuh"

#include <iostream>
#include <unordered_map>

extern "C"
{

void hello_world();

// TODO: implement initial pose input
int create_frame(
	int width, int height, int level,
	float fx, float fy, float cx, float cy);
int create_object(
	int width, int height, int depth,
	float center_x, float center_y, float center_z, float voxel_size);
int create_camera(
	int width, int height,
	float fx, float fy, float cx, float cy);

void delete_frame(int id);
void delete_object(int id);
void delete_camera(int id);

void frame_preprocess(int frame_id, const float* depth_map);
void frame_mask(int frame_id, const bool* mask);
void frame_clear(int frame_id);
void object_integrate_frame(int object_id, int frame_id);
void camera_raycast_object(int camera_id, int object_id);
void camera_track_frame(int camera_id, int frame_id);
void camera_clear(int camera_id);

// TODO: implement frame level input
const float* get_frame_depth_map(int frame_id);
const float* get_frame_vertex_map(int frame_id);
const float* get_frame_normal_map(int frame_id);
const float* get_object_tsdf_volume(int object_id);
const float* get_object_weight_volume(int object_id);
const float* get_object_binomial_volume(int object_id);
const float* get_object_polygon_volume(int object_id);
const float* get_camera_depth_map(int camera_id);
const float* get_camera_vertex_map(int camera_id);
const float* get_camera_normal_map(int camera_id);
const int* get_camera_instance_map(int camera_id);
const float* get_camera_pose(int camera_id);
float get_camera_track_error(int camera_id);

}
