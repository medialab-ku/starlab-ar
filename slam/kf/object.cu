#include "object.cuh"
#include "table.cuh"

namespace kf
{
	__constant__ KF_MCT_NODE_TYPE_DEVICE;
	__constant__ KF_MCT_EDGE_TYPE_DEVICE;
	__constant__ KF_MCT_POLY_TYPE_DEVICE;

	bool CObject::s_bInitMarchingCube = false;

	CObject::CObject(
		unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth,
		float fCenterX, float fCenterY, float fCenterZ, float fVoxelSize,
		TInstance instance) :
		m_uiWidth(uiWidth), m_uiHeight(uiHeight), m_uiDepth(uiDepth),
		m_center(fCenterX, fCenterY, fCenterZ),
		m_fVoxelSize(fVoxelSize),
		m_instance(instance),
		m_positionVolume(uiWidth, uiHeight, uiDepth),
		m_tsdfVolume(uiWidth, uiHeight, uiDepth),
		m_weightVolume(uiWidth, uiHeight, uiDepth),
		m_countVolumeA(uiWidth, uiHeight, uiDepth),
		m_countVolumeB(uiWidth, uiHeight, uiDepth),
		m_binomialVolume(uiWidth, uiHeight, uiDepth),
		m_polygonVolume(uiWidth, uiHeight, uiDepth)
	{
		// allocate volume memory
		m_positionVolume.malloc();
		m_tsdfVolume.malloc();
		m_weightVolume.malloc();
		m_countVolumeA.malloc();
		m_countVolumeB.malloc();
		m_binomialVolume.malloc();
		m_polygonVolume.malloc();

		// reset volume data
		resetVolume();

		// initialize marching cube tables
		if (!s_bInitMarchingCube)
		{
			cudaMemcpyToSymbol(KF_MCT_NODE_NAME_DEVICE, KF_MCT_NODE_NAME_HOST, KF_MCT_NODE_SIZE);
			cudaMemcpyToSymbol(KF_MCT_EDGE_NAME_DEVICE, KF_MCT_EDGE_NAME_HOST, KF_MCT_EDGE_SIZE);
			cudaMemcpyToSymbol(KF_MCT_POLY_NAME_DEVICE, KF_MCT_POLY_NAME_HOST, KF_MCT_POLY_SIZE);

			s_bInitMarchingCube = true;
		}
	}

	CObject::~CObject()
	{
		// free volume memory
		m_positionVolume.free();
		m_tsdfVolume.free();
		m_weightVolume.free();
		m_countVolumeA.free();
		m_countVolumeB.free();
		m_binomialVolume.free();
		m_polygonVolume.free();
	}

	void CObject::integrate(const CFrame& frame) const
	{
#if KF_DEBUG
		// DEBUG: timer to measure fusion time
		CTimer timer("CObject::integrate");
#endif

		// update volume data
		updateVolume(frame);
	}

	__global__ void kernelResetVolume(
		TPositionVolume positionVolume, TTsdfVolume tsdfVolume, TWeightVolume weightVolume,
		TCountVolumeA countVolumeA, TCountVolumeB countVolumeB,
		TPosition center, float voxelSize,
		unsigned int width, unsigned int height, unsigned int depth)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
		if (x < width && y < height && z < depth)
		{
			// compute origin
			TPosition origin(
				center.x() - width * voxelSize / 2,
				center.y() - height * voxelSize / 2,
				center.z() - depth * voxelSize / 2);

			// compute position
			TPosition position(
				origin.x() + x * voxelSize,
				origin.y() + y * voxelSize,
				origin.z() + z * voxelSize);
			positionVolume(x, y, z) = position;

			// initialize TSDF and weight
			// NOTE: initialize TSDF to out of SDF RANGE
			tsdfVolume(x, y, z) = 2 * KF_FUSE_SDF_RANGE;
			weightVolume(x, y, z) = 0;

			// initialize count
			countVolumeA(x, y, z) = 0;
			countVolumeB(x, y, z) = 0;
		}
	}

	void CObject::resetVolume() const
	{
		// reset volume (CUDA)
		kernelResetVolume<<<m_positionVolume.dimGrid, m_positionVolume.dimBlock>>>(
			m_positionVolume, m_tsdfVolume, m_weightVolume,
			m_countVolumeA, m_countVolumeB,
			m_center, m_fVoxelSize,
			m_uiWidth, m_uiHeight, m_uiDepth);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CObject::resetVolume - failed to reset volume");
		}

#if KF_DEBUG
		// DEBUG: download position volume
		m_positionVolume.download();

		// DEBUG: download TSDF volume
		m_tsdfVolume.download();

		// DEBUG: download weight volume
		m_weightVolume.download();
#endif
	}

	__global__ void kernelUpdateVolume(
		TPositionVolume positionVolume, TTsdfVolume tsdfVolume, TWeightVolume weightVolume,
		TCountVolumeA countVolumeA, TCountVolumeB countVolumeB, TBinomialVolume binomialVolume,
		TDepthMap depthMap, TNormalMap normalMap, TValidityMap validityMap, TMaskMap maskMap,
		float voxelSize,
		SIntrinsic intrinsic, TPose poseInv,
		unsigned int volumeWidth, unsigned int volumeHeight, unsigned int volumeDepth,
		unsigned int mapWidth, unsigned int mapHeight)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
		if (x < volumeWidth && y < volumeHeight && z < volumeDepth)
		{
			// compute position in camera space
			TPosition position =
				poseInv.block<3, 3>(0, 0) * positionVolume(x, y, z) +
				poseInv.block<3, 1>(0, 3);

			// project position
			int x_ = std::floor(position.x() / position.z() * intrinsic.fx + intrinsic.cx);
			int y_ = std::floor(position.y() / position.z() * intrinsic.fy + intrinsic.cy);
			if (x_ >= 0 && x_ < mapWidth && y_ >= 0 && y_ < mapHeight)
			{
				// check validity and mask
				if (validityMap(x_, y_))
				{
					// compute TSDF and Weight
					float sdf = (depthMap(x_, y_) - position.z()) / voxelSize;
					if (sdf > -KF_FUSE_SDF_RANGE)
					{
						// get old TSDF and weight
						TTsdf& tsdf = tsdfVolume(x, y, z);
						TWeight& weight = weightVolume(x, y, z);

						// compute new TSDF and weight
						TTsdf tsdf_ = (sdf < KF_FUSE_SDF_RANGE ? sdf : KF_FUSE_SDF_RANGE);
						TWeight weight_ = KF_FUSE_WEIGHT_SCALE * normalMap(x_, y_).dot(position.normalized());

						// check weight threshold
						if (weight_ > KF_FUSE_WEIGHT_THRESH)
						{
							// update TSDF and weight
							tsdf = (tsdf * weight + tsdf_ * weight_) / (weight + weight_);
							weight = (weight + weight_ < KF_FUSE_WEIGHT_MAX ? weight + weight_ : KF_FUSE_WEIGHT_MAX);

							// increase count by mask
							TCount& countA = countVolumeA(x, y, z);
							TCount& countB = countVolumeB(x, y, z);
							if (maskMap(x_, y_))
							{
								countA += 1;
							}
							else
							{
								countB += 1;
							}

							// update binomial probability
							binomialVolume(x, y, z) = static_cast<TBinomial>(countA) / (countA + countB);
						}
					}
				}
			}
		}
	}

	void CObject::updateVolume(const CFrame& frame) const
	{
		// get 0 level image
		const CImage& image = frame.pyramid.at(0);

		// update volume (CUDA)
		kernelUpdateVolume<<<m_tsdfVolume.dimGrid, m_tsdfVolume.dimBlock>>>(
			m_positionVolume, m_tsdfVolume, m_weightVolume,
			m_countVolumeA, m_countVolumeB, m_binomialVolume,
			image.depthMap, image.normalMapF, image.validityMap, frame.maskMap,
			m_fVoxelSize,
			image.intrinsic, frame.pose.inverse(),
			m_uiWidth, m_uiHeight, m_uiDepth,
			image.width, image.height);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CObject::updateVolume - failed to update volume");
		}

#if KF_DEBUG
		// DEBUG: download TSDF volume
		m_tsdfVolume.download();

		// DEBUG: download weight volume
		m_weightVolume.download();
#endif
	}

	__global__ void kernelMarchingCube(
		TPositionVolume positionVolume, TTsdfVolume tsdfVolume,
		TBinomialVolume binomialVolume, TPolygonVolume polygonVolume,
		unsigned int width, unsigned int height, unsigned int depth)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
		if (x + 1 < width && y + 1 < height && z + 1 < depth)
		{
			// compute polygon index
			unsigned int indexPoly = 0x0;
			for (unsigned int i = 0; i < 8; i++)
			{
				// shift polygon index bits
				indexPoly <<= 1;

				// check valid TSDF and binomial value
				auto tsdf = tsdfVolume(x + KF_MCT_NODE_NAME_DEVICE[i][0],
									   y + KF_MCT_NODE_NAME_DEVICE[i][1],
									   z + KF_MCT_NODE_NAME_DEVICE[i][2]);
				auto binomial = binomialVolume(x + KF_MCT_NODE_NAME_DEVICE[i][0],
											   y + KF_MCT_NODE_NAME_DEVICE[i][1],
											   z + KF_MCT_NODE_NAME_DEVICE[i][2]);
				if (tsdf > KF_FUSE_SDF_RANGE || binomial <= 0.5f)
				{
					// invalid case - zero index
					indexPoly = 0x0;
					break;
				}
				else
				{
					// set polygon index bit
					if (tsdf > 0.0f)
						indexPoly |= 0x1;
				}
			}

			// compute polygons(vertices)
			for (unsigned int i = 0; i < 5 * 3; i++)
			{
				// get edge index
				auto indexEdge = KF_MCT_POLY_NAME_DEVICE[indexPoly][i];

				// check valid edge index
				if (indexEdge == 255)
				{
					// invalid case - zero vertex
					polygonVolume(x, y, z, i).setZero();
				}
				else
				{
					// get index nodes
					auto indexNodeP = KF_MCT_EDGE_NAME_DEVICE[indexEdge][0];
					auto indexNodeQ = KF_MCT_EDGE_NAME_DEVICE[indexEdge][1];

					// get node positions
					auto positionP = positionVolume(x + KF_MCT_NODE_NAME_DEVICE[indexNodeP][0],
													y + KF_MCT_NODE_NAME_DEVICE[indexNodeP][1],
													z + KF_MCT_NODE_NAME_DEVICE[indexNodeP][2]);
					auto positionQ = positionVolume(x + KF_MCT_NODE_NAME_DEVICE[indexNodeQ][0],
													y + KF_MCT_NODE_NAME_DEVICE[indexNodeQ][1],
													z + KF_MCT_NODE_NAME_DEVICE[indexNodeQ][2]);

					// get node TSDF
					auto tsdfP = tsdfVolume(x + KF_MCT_NODE_NAME_DEVICE[indexNodeP][0],
											y + KF_MCT_NODE_NAME_DEVICE[indexNodeP][1],
											z + KF_MCT_NODE_NAME_DEVICE[indexNodeP][2]);
					auto tsdfQ = tsdfVolume(x + KF_MCT_NODE_NAME_DEVICE[indexNodeQ][0],
											y + KF_MCT_NODE_NAME_DEVICE[indexNodeQ][1],
											z + KF_MCT_NODE_NAME_DEVICE[indexNodeQ][2]);

					// apply interpolation to compute polygon vertex
					auto distP = std::abs(tsdfP);
					auto distQ = std::abs(tsdfQ);
					polygonVolume(x, y, z, i) = (positionP * distQ + positionQ * distP) / (distP + distQ);
				}
			}
		}
	}

	void CObject::applyMarchingCube() const
	{
#if KF_DEBUG
		// DEBUG: timer to measure Marching Cube time
		CTimer timer("CObject::applyMarchingCube");
#endif

		// apply Marching Cube (CUDA)
		kernelMarchingCube<<<m_positionVolume.dimGrid, m_positionVolume.dimBlock>>>(
			m_positionVolume, m_tsdfVolume,
			m_binomialVolume, m_polygonVolume,
			m_uiWidth, m_uiHeight, m_uiDepth);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CObject::applyMarchingCube - failed to apply Marching Cube");
		}

#if KF_DEBUG
		// DEBUG: download polygon volume
		m_polygonVolume.download();
#endif
	}
}
