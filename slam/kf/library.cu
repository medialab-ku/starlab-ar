#include "library.cuh"

// frame, object, camera counter
int frame_count = 0;
int object_count = 0;
int camera_count = 0;

// frame, object, camera hash map
std::unordered_map<int, kf::CFrame> frames;
std::unordered_map<int, kf::CObject> objects;
std::unordered_map<int, kf::CCamera> cameras;

void hello_world()
{
	// print message
	std::cout << "Hello, World!" << std::endl;
}

int create_frame(
	int width, int height, int level,
	float fx, float fy, float cx, float cy)
{
	// increase frame count
	frame_count++;

	// set intrinsic
	kf::SIntrinsic intrinsic{ fx, fy, cx, cy };

	// create frame
	frames.emplace(std::piecewise_construct,
		std::forward_as_tuple(frame_count),
		std::forward_as_tuple(width, height, level, intrinsic));

	// return ID by frame count
	return frame_count;
}

int create_object(
	int width, int height, int depth,
	float center_x, float center_y, float center_z, float voxel_size)
{
	// increase object count
	object_count++;

	// create object
	objects.emplace(std::piecewise_construct,
		std::forward_as_tuple(object_count),
		std::forward_as_tuple(width, height, depth,
			center_x, center_y, center_z, voxel_size,
			object_count));

	// return ID by object count
	return object_count;
}

int create_camera(
	int width, int height,
	float fx, float fy, float cx, float cy)
{
	// increase camera count
	camera_count++;

	// set intrinsic
	kf::SIntrinsic intrinsic{ fx, fy, cx, cy };

	// create camera
	cameras.emplace(std::piecewise_construct,
		std::forward_as_tuple(camera_count),
		std::forward_as_tuple(width, height, intrinsic));

	// return ID by camera count
	return camera_count;
}

void delete_frame(int id)
{
	// check frame exist
	if (frames.find(id) == frames.end())
	{
		std::cerr << "delete_frame - invalid frame ID" << std::endl;
		return;
	}

	// delete frame
	frames.erase(id);
}

void delete_object(int id)
{
	// check object exist
	if (objects.find(id) == objects.end())
	{
		std::cerr << "delete_object - invalid object ID" << std::endl;
		return;
	}

	// delete object
	objects.erase(id);
}

void delete_camera(int id)
{
	// check camera exist
	if (cameras.find(id) == cameras.end())
	{
		std::cerr << "delete_camera - invalid camera ID" << std::endl;
		return;
	}

	// delete camera
	cameras.erase(id);
}

void frame_preprocess(int frame_id, const float* depth_map)
{
	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "frame_preprocess - invalid frame ID" << std::endl;
		return;
	}

	// apply frame preprocess
	frames.at(frame_id).preprocess(depth_map);
}

void frame_mask(int frame_id, const bool* mask)
{
	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "frame_mask - invalid frame ID" << std::endl;
		return;
	}

	// set frame mask
	frames.at(frame_id).mask(mask);
}

void frame_clear(int frame_id)
{
	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "frame_clear - invalid frame ID" << std::endl;
		return;
	}

	// clear frame mask
	frames.at(frame_id).clear();
}

void object_integrate_frame(int object_id, int frame_id)
{
	// check object exist
	if (objects.find(object_id) == objects.end())
	{
		std::cerr << "object_integrate_frame - invalid object ID" << std::endl;
		return;
	}

	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "object_integrate_frame - invalid frame ID" << std::endl;
		return;
	}

	// apply object integrate frame
	objects.at(object_id).integrate(frames.at(frame_id));
}

void camera_raycast_object(int camera_id, int object_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "camera_raycast_object - invalid camera ID" << std::endl;
		return;
	}

	// check object exist
	if (objects.find(object_id) == objects.end())
	{
		std::cerr << "camera_raycast_object - invalid object ID" << std::endl;
		return;
	}

	// apply camera ray-cast object
	cameras.at(camera_id).raycast(objects.at(object_id));
}

void camera_track_frame(int camera_id, int frame_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "camera_track_frame - invalid camera ID" << std::endl;
		return;
	}

	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "camera_track_frame - invalid frame ID" << std::endl;
		return;
	}

	// apply camera track frame
	cameras.at(camera_id).track(frames.at(frame_id));
}

void camera_clear(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "camera_clear - invalid camera ID" << std::endl;
		return;
	}

	// apply camera clear
	cameras.at(camera_id).clear();
}

const float* get_frame_depth_map(int frame_id)
{
	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "get_frame_depth_map - invalid frame ID" << std::endl;
		return nullptr;
	}

	// download frame depth map
	frames.at(frame_id).pyramid.at(0).depthMap.download();

	// return frame depth map
	return reinterpret_cast<const float*>(frames.at(frame_id).pyramid.at(0).depthMap.data());
}

const float* get_frame_vertex_map(int frame_id)
{
	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "get_frame_vertex_map - invalid frame ID" << std::endl;
		return nullptr;
	}

	// download frame vertex map
	frames.at(frame_id).pyramid.at(0).vertexMap.download();

	// return frame vertex map
	return reinterpret_cast<const float*>(frames.at(frame_id).pyramid.at(0).vertexMap.data());
}

const float* get_frame_normal_map(int frame_id)
{
	// check frame exist
	if (frames.find(frame_id) == frames.end())
	{
		std::cerr << "get_frame_normal_map - invalid frame ID" << std::endl;
		return nullptr;
	}

	// download frame normal map
	frames.at(frame_id).pyramid.at(0).normalMapF.download();

	// return frame normal map
	return reinterpret_cast<const float*>(frames.at(frame_id).pyramid.at(0).normalMapF.data());
}

const float* get_object_tsdf_volume(int object_id)
{
	// check object exist
	if (objects.find(object_id) == objects.end())
	{
		std::cerr << "get_object_tsdf_volume - invalid object ID" << std::endl;
		return nullptr;
	}

	// download object TSDF volume
	objects.at(object_id).tsdfVolume.download();

	// return object TSDF volume
	return reinterpret_cast<const float*>(objects.at(object_id).tsdfVolume.data());
}

const float* get_object_weight_volume(int object_id)
{
	// check object exist
	if (objects.find(object_id) == objects.end())
	{
		std::cerr << "get_object_weight_volume - invalid object ID" << std::endl;
		return nullptr;
	}

	// download object weight volume
	objects.at(object_id).weightVolume.download();

	// return object weight volume
	return reinterpret_cast<const float*>(objects.at(object_id).weightVolume.data());
}

const float* get_object_binomial_volume(int object_id)
{
	// check object exist
	if (objects.find(object_id) == objects.end())
	{
		std::cerr << "get_object_binomial_volume - invalid object ID" << std::endl;
		return nullptr;
	}

	// download object binomial volume
	objects.at(object_id).binomialVolume.download();

	// return object binomial volume
	return reinterpret_cast<const float*>(objects.at(object_id).binomialVolume.data());
}

const float* get_object_polygon_volume(int object_id)
{
	// check object exist
	if (objects.find(object_id) == objects.end())
	{
		std::cerr << "get_object_polygon_volume - invalid object ID" << std::endl;
		return nullptr;
	}

	// apply Marching Cube
	objects.at(object_id).applyMarchingCube();

	// download object polygon volume
	objects.at(object_id).polygonVolume.download();

	// return object polygon volume
	return reinterpret_cast<const float*>(objects.at(object_id).polygonVolume.data());
}

const float* get_camera_depth_map(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "get_camera_depth_map - invalid camera ID" << std::endl;
		return nullptr;
	}

	// download camera depth map
	cameras.at(camera_id).depthMap.download();

	// return camera depth map
	return reinterpret_cast<const float*>(cameras.at(camera_id).depthMap.data());
}

const float* get_camera_vertex_map(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "get_camera_vertex_map - invalid camera ID" << std::endl;
		return nullptr;
	}

	// download camera vertex map
	cameras.at(camera_id).vertexMap.download();

	// return camera vertex map
	return reinterpret_cast<const float*>(cameras.at(camera_id).vertexMap.data());
}

const float* get_camera_normal_map(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "get_camera_normal_map - invalid camera ID" << std::endl;
		return nullptr;
	}

	// download camera normal map
	cameras.at(camera_id).normalMap.download();

	// return camera normal map
	return reinterpret_cast<const float*>(cameras.at(camera_id).normalMap.data());
}

const int* get_camera_instance_map(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "get_camera_instance_map - invalid camera ID" << std::endl;
		return nullptr;
	}

	// download camera instance map
	cameras.at(camera_id).instanceMap.download();

	// return camera instance map
	return reinterpret_cast<const int*>(cameras.at(camera_id).instanceMap.data());
}

const float* get_camera_pose(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "get_camera_pose - invalid camera ID" << std::endl;
		return nullptr;
	}

	// return camera pose
	return reinterpret_cast<const float*>(cameras.at(camera_id).pose.data());
}

float get_camera_track_error(int camera_id)
{
	// check camera exist
	if (cameras.find(camera_id) == cameras.end())
	{
		std::cerr << "get_camera_track_error - invalid camera ID" << std::endl;
		return -1.0f;
	}

	// return camera track error
	return cameras.at(camera_id).trackError;
}
