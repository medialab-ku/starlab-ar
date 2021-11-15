#include "camera.cuh"

namespace kf
{
	CCamera::CCamera(
		unsigned int uiWidth, unsigned int uiHeight,
		SIntrinsic intrinsic, TPose pose) :
		m_uiWidth(uiWidth), m_uiHeight(uiHeight),
		m_intrinsic(intrinsic), m_pose(pose),
		m_depthMap(uiWidth, uiHeight),
		m_vertexMap(uiWidth, uiHeight),
		m_normalMap(uiWidth, uiHeight),
		m_validityMap(uiWidth, uiHeight),
		m_instanceMap(uiWidth, uiHeight),
		m_choleskyMapA(uiWidth, uiHeight),
		m_choleskyMapB(uiWidth, uiHeight),
		m_errorMap(uiWidth, uiHeight),
		m_countMap(uiWidth, uiHeight)
	{
		// allocate map memory
		m_depthMap.malloc();
		m_vertexMap.malloc();
		m_normalMap.malloc();
		m_validityMap.malloc();
		m_instanceMap.malloc();
		m_choleskyMapA.malloc();
		m_choleskyMapB.malloc();
		m_errorMap.malloc();
		m_countMap.malloc();

		// clear map data
		clear();
	}

	CCamera::~CCamera()
	{
		// free map memory
		m_depthMap.free();
		m_vertexMap.free();
		m_normalMap.free();
		m_validityMap.free();
		m_instanceMap.free();
		m_choleskyMapA.free();
		m_choleskyMapB.free();
		m_errorMap.free();
		m_countMap.free();
	}

	void CCamera::raycast(const CObject& object) const
	{
#if KF_DEBUG
		// DEBUG: timer to measure ray-casting time
		CTimer timer("CCamera::raycast");
#endif

		// ray-cast object
		raycastObject(object);
	}

	void CCamera::track(CFrame& frame)
	{
#if KF_DEBUG
		// DEBUG: timer to measure tracking time
		CTimer timer("CCamera::track");
#endif

		// track frame
		trackFrame(frame);

		// update camera pose to tracked frame pose
		m_pose = frame.pose;

		// download sum of ICP error and valid ICP count
		m_errorMap.download(1);
		m_countMap.download(1);

		// compute average ICP tracking error
		m_fTrackError = m_errorMap.data()[0] / m_countMap.data()[0];
	}

	void CCamera::clear() const
	{
		// clear map data
		clearMap();
	}

	__global__ void kernelRaycastObject(
		TDepthMap depthMap, TVertexMap vertexMap, TNormalMap normalMap,
		TValidityMap validityMap, TInstanceMap instanceMap,
		TPositionVolume positionVolume, TTsdfVolume tsdfVolume, TBinomialVolume binomialVolume,
		SIntrinsic intrinsic, TPose pose,
		float voxelSize, TInstance instance,
		unsigned int mapWidth, unsigned int mapHeight,
		unsigned int volumeWidth, unsigned int volumeHeight, unsigned int volumeDepth)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < mapWidth && y < mapHeight)
		{
			// compute eye and ray in world space
			TPosition eye = pose.block<3, 1>(0, 3);
			TVector ray = pose.block<3, 3>(0, 0) * TVector(
				(x - intrinsic.cx) / intrinsic.fx,
				(y - intrinsic.cy) / intrinsic.fy,
				1).normalized();

			// get min and max value of ray step and position
			float stepMin = KF_RC_NEAR_Z / ray.z();
			float stepMax = KF_RC_FAR_Z / ray.z();
			TPosition positionMin = positionVolume(0, 0, 0);
			TPosition positionMax = positionVolume(volumeWidth - 1, volumeHeight - 1, volumeDepth - 1);

			// initialize ray step and position
			float step = stepMin;
			float step_ = stepMin + KF_RC_STEP_SLOW * voxelSize;
			TPosition position = eye + step * ray;
			TPosition position_ = eye + step_ * ray;

			// initialize voxel indices and TSDF
			int x_ = -1;
			int y_ = -1;
			int z_ = -1;
			TTsdf tsdf = -1;
			TTsdf tsdf_ = -1;

			// propagate ray
			while (step_ < stepMax)
			{
				// check ray is inside volume
				if (position_.x() > positionMin.x() && position_.x() < positionMax.x() &&
					position_.y() > positionMin.y() && position_.y() < positionMax.y() &&
					position_.z() > positionMin.z() && position_.z() < positionMax.z())
				{
					// compute voxel indices
					x_ = std::floor((position_.x() - positionMin.x()) / voxelSize);
					y_ = std::floor((position_.y() - positionMin.y()) / voxelSize);
					z_ = std::floor((position_.z() - positionMin.z()) / voxelSize);

					// check voxel indices are not zeros
					// (for computing normal vector later)
					if (x_ > 0 && y_ > 0 && z_ > 0)
					{
						// update TSDF
						tsdf = tsdf_;
						tsdf_ = tsdfVolume(x_, y_, z_);

						// check ray hit surface
						if (tsdf > 0 && tsdf_ < 0)
						{
							// check binomial probability
							TBinomial binomial = binomialVolume(x_, y_, z_);
							if (binomial > 0.5)
							{
								break;
							}
						}
					}

					// update ray step and position (fast)
					step = step_;
					step_ += KF_RC_STEP_SLOW * voxelSize;
					position = position_;
					position_ += KF_RC_STEP_SLOW * voxelSize * ray;
				}
				else
				{
					// update ray step and position (slow)
					step = step_;
					step_ += KF_RC_STEP_FAST * voxelSize;
					position = position_;
					position_ += KF_RC_STEP_FAST * voxelSize * ray;
				}
			}

			// check ray hit surface before
			if (step_ < stepMax && tsdf > 0 && tsdf_ < 0)
			{
				// compute linear interpolation terms
				float lerp = -tsdf_ / (tsdf - tsdf_);
				float lerp_ = tsdf / (tsdf - tsdf_);

				// compute depth, vertex, normal, validity, instance
				depthMap(x, y) = (step * lerp + step_ * lerp_) * ray.z();
				vertexMap(x, y) = position * lerp + position_ * lerp_;
				normalMap(x, y) = -TNormal(
					tsdf_ - tsdfVolume(x_ - 1, y_ - 0, z_ - 0),
					tsdf_ - tsdfVolume(x_ - 0, y_ - 1, z_ - 0),
					tsdf_ - tsdfVolume(x_ - 0, y_ - 0, z_ - 1)).normalized();
				validityMap(x, y) = true;
				instanceMap(x, y) = instance;
			}
		}
	}

	void CCamera::raycastObject(const CObject& object) const
	{
		// ray-cast object (CUDA)
		kernelRaycastObject<<<m_vertexMap.dimGrid, m_vertexMap.dimBlock>>>(
			m_depthMap, m_vertexMap, m_normalMap,
			m_validityMap, m_instanceMap,
			object.positionVolume, object.tsdfVolume, object.binomialVolume,
			m_intrinsic, m_pose,
			object.voxelSize, object.instance,
			m_uiWidth, m_uiHeight,
			object.width, object.height, object.depth);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CCamera::raycastObject - failed to ray-cast object");
		}

#if KF_DEBUG
		// DEBUG: download vertex map (w/ ray-cast)
		m_vertexMap.download();

		// DEBUG: download normal map (w/ ray-cast)
		m_normalMap.download();

		// DEBUG: download validity map (w/ ray-cast)
		m_validityMap.download();
#endif
	}

	__global__ void kernelTrackFrame(
		TVertexMap vertexMap_, TNormalMap normalMap_, TValidityMap validityMap_,
		TCholeskyMapA choleskyMapA, TCholeskyMapB choleskyMapB, TErrorMap errorMap, TCountMap countMap,
		TVertexMap vertexMap, TNormalMap normalMap, TValidityMap validityMap,
		SIntrinsic intrinsic, TPose poseInv, TPose poseTrack,
		unsigned int width_, unsigned int height_,
		unsigned int width, unsigned int height)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			// initialize cholesky terms to zeros
			choleskyMapA(x, y).setZero();
			choleskyMapB(x, y).setZero();

			// initialize error and count to zeros
			errorMap(x, y) = 0;
			countMap(x, y) = 0;

			// compute vertex and normal in world space
			TVertex vertex = poseTrack.block<3, 3>(0, 0) * vertexMap(x, y) +
				poseTrack.block<3, 1>(0, 3);
			TNormal normal = poseTrack.block<3, 3>(0, 0) * normalMap(x, y);

			// project vertex to image space (w/ ray-cast)
			TPosition position_ = poseInv.block<3, 3>(0, 0) * vertex +
				poseInv.block<3, 1>(0, 3);
			int x_ = std::floor(position_.x() / position_.z() * intrinsic.fx + intrinsic.cx);
			int y_ = std::floor(position_.y() / position_.z() * intrinsic.fy + intrinsic.cy);
			if (x_ >= 0 && x_ < width_ && y_ >= 0 && y_ < height_)
			{
				// get vertex and normal in world space (w/ ray-cast)
				// (vertex and normal already in world space)
				TVertex vertex_ = vertexMap_(x_, y_);
				TNormal normal_ = normalMap_(x_, y_);

				// check validity
				if (validityMap(x, y) && validityMap_(x_, y_))
				{
					// check distance and angle threshold
					// (compare distance by norm and angle by dot product)
					float threshDist = (vertex - vertex_).norm();
					float threshAngle = normal.dot(normal_);
					if (threshDist < KF_ICP_EPS_DIST && threshAngle > KF_ICP_EPS_ANGLE)
					{
						// compute jacobian and residual
						TJacobian jacobianT;
						jacobianT << -vertex.cross(normal_), normal_;
						TResidual residual = normal_.dot(vertex_ - vertex);

						// compute cholesky terms
						choleskyMapA(x, y) = jacobianT * jacobianT.transpose();
						choleskyMapB(x, y) = jacobianT * residual;

						// set ICP error and count
						errorMap(x, y) = std::abs(residual);
						countMap(x, y) = 1;
					}
				}
			}
		}

		// synchronize all threads in block
		__syncthreads();

		// apply 1st stage reduce sum (inside block)
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			for (unsigned int x__ = blockDim.x * blockIdx.x; x__ < blockDim.x * (blockIdx.x + 1); x__++)
			{
				for (unsigned int y__ = blockDim.y * blockIdx.y; y__ < blockDim.y * (blockIdx.y + 1); y__++)
				{
					if (x__ < width && y__ < height)
					{
						if (x == x__ && y == y__)
						{
							continue;
						}

						choleskyMapA(x, y) += choleskyMapA(x__, y__);
						choleskyMapB(x, y) += choleskyMapB(x__, y__);

						errorMap(x, y) += errorMap(x__, y__);
						countMap(x, y) += countMap(x__, y__);
					}
				}
			}
		}
	}

	__global__ void kernelReduceSum(
		TCholeskyMapA choleskyMapA, TCholeskyMapB choleskyMapB, TErrorMap errorMap, TCountMap countMap,
		unsigned int blockWidth, unsigned int blockHeight,
		unsigned int mapWidth, unsigned int mapHeight)
	{
		// apply 2nd stage reduce sum (entire map)
		// (already reduced by block before)
		for (unsigned int x = 0; x < mapWidth; x += blockWidth)
		{
			for (unsigned int y = 0; y < mapHeight; y += blockHeight)
			{
				if (x == 0 && y == 0)
				{
					continue;
				}

				choleskyMapA(0, 0) += choleskyMapA(x, y);
				choleskyMapB(0, 0) += choleskyMapB(x, y);

				errorMap(0, 0) += errorMap(x, y);
				countMap(0, 0) += countMap(x, y);
			}
		}
	}

	void CCamera::trackFrame(CFrame& frame) const
	{
		// initialize tracking pose
		TPose poseInv = frame.pose.inverse();
		TPose poseTrack = frame.pose;

		// apply ICP method to track frame with camera
		unsigned int uiaIteration[] = KF_ICP_ITER_NUM;
		for (unsigned int l = frame.pyramid.size(); l-- > 0; )
		{
			const CImage& image = frame.pyramid.at(l);
			for (unsigned int i = 0; i < uiaIteration[l]; i++)
			{
				// track frame (w/ 1st stage reduce sum) (CUDA)
				kernelTrackFrame<<<image.vertexMap.dimGrid, image.vertexMap.dimBlock>>>(
					m_vertexMap, m_normalMap, m_validityMap,
					m_choleskyMapA, m_choleskyMapB, m_errorMap, m_countMap,
					image.vertexMap, image.normalMapF, image.validityMap,
					m_intrinsic, poseInv, poseTrack,
					m_uiWidth, m_uiHeight,
					image.width, image.height);
				if (cudaDeviceSynchronize() != cudaSuccess)
				{
					throw std::runtime_error("kf::CCamera::trackFrame - failed to track frame");
				}

				// apply 2nd stage reduce sum (CUDA)
				kernelReduceSum<<<1, 1>>>(
					m_choleskyMapA, m_choleskyMapB, m_errorMap, m_countMap,
					image.vertexMap.dimBlock.x, image.vertexMap.dimBlock.y,
					frame.width, frame.height);
				if (cudaDeviceSynchronize() != cudaSuccess)
				{
					throw std::runtime_error("kf::CCamera::trackFrame - failed to reduce sum");
				}

				// download sum of cholesky term
				m_choleskyMapA.download(1);
				m_choleskyMapB.download(1);

				// apply cholesky decomposition
				TCholeskyA a = m_choleskyMapA.data()[0];
				TCholeskyB b = m_choleskyMapB.data()[0];
				TCholeskyX x = a.llt().solve(b);

				// apply incremental tracking pose
				TPose poseInc;
				poseInc << 1, x(2), -x(1), x(3),
					-x(2), 1, x(0), x(4),
					x(1), -x(0), 1, x(5),
					0, 0, 0, 1;
				poseTrack = poseInc * poseTrack;
			}
		}

		// update frame pose
		frame.pose = poseTrack;
	}

	__global__ void kernelClearMap(
		TDepthMap depthMap, TVertexMap vertexMap, TNormalMap normalMap,
		TValidityMap validityMap, TInstanceMap instanceMap,
		unsigned int width, unsigned int height)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			// clear map data to invalid values
			depthMap(x, y) = KF_RC_FAR_Z; // set far Z depth
			vertexMap(x, y).setZero();
			normalMap(x, y).setZero();
			validityMap(x, y) = false;
			instanceMap(x, y) = -1;
		}
	}

	void CCamera::clearMap() const
	{
		// clear map (CUDA)
		kernelClearMap<<<m_depthMap.dimGrid, m_depthMap.dimBlock>>>(
			m_depthMap, m_vertexMap, m_normalMap,
			m_validityMap, m_instanceMap,
			m_uiWidth, m_uiHeight);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CCamera::clearMap - failed to clear map");
		}
	}
}
