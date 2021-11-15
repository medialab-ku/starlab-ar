#include "frame.cuh"

namespace kf
{
	CFrame::CFrame(
		unsigned int uiWidth, unsigned int uiHeight, unsigned int uiLevel,
		SIntrinsic intrinsic, TPose pose) :
		m_uiWidth(uiWidth), m_uiHeight(uiHeight), m_uiLevel(uiLevel),
		m_intrinsic(intrinsic), m_pose(pose),
		m_maskMap(uiWidth, uiHeight)
	{
		// allocate memory to create image pyramid
		m_pyramid.reserve(uiLevel);

		// create image pyramid
		for (unsigned int l = 0; l < uiLevel; l++)
		{
			// create image
			m_pyramid.emplace_back(uiWidth, uiHeight, intrinsic);

			// downscale width, height, intrinsic
			uiWidth /= KF_SS_DOWN_SCALE;
			uiHeight /= KF_SS_DOWN_SCALE;
			intrinsic.fx /= KF_SS_DOWN_SCALE;
			intrinsic.fy /= KF_SS_DOWN_SCALE;
			intrinsic.cx /= KF_SS_DOWN_SCALE;
			intrinsic.cy /= KF_SS_DOWN_SCALE;
		}

		// allocate mask memory
		m_maskMap.malloc();

		// clear mask
		clear();
	}

	CFrame::~CFrame()
	{
		// free mask memory
		m_maskMap.free();
	}

	void CFrame::preprocess(const void* pDepth) const
	{
#if KF_DEBUG
		// DEBUG: timer to measure preprocess time
		CTimer timer("CFrame::preprocess");
#endif

		// set input depth data to 0 level image
		m_pyramid.at(0).input(static_cast<const TDepth*>(pDepth));

		// apply bilateral filter to 0 level image
		m_pyramid.at(0).filter();

		// apply sub-sampling
		applySubSample();

		// compute vertex, normal, validity for each image
		for (const CImage& image : m_pyramid)
		{
			image.compute();
		}
	}

	void CFrame::mask(const void* pMask) const
	{
		// check input mask data
		if (!pMask)
		{
			throw std::runtime_error("kf::CFrame::mask - no mask data input");
		}

		// copy & upload mask data
		std::copy_n(static_cast<const TMask*>(pMask),
			m_maskMap.size(), m_maskMap.data());
		m_maskMap.upload();
	}

	void CFrame::clear() const
	{
		// clear mask data
		clearMask();
	}

	__global__ void kernelSubSample(
		TDepthMap depthMap_, TDepthMap depthMap,
		unsigned int width, unsigned int height)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			// check valid depth
			TDepth depth = depthMap_(KF_SS_DOWN_SCALE * x, KF_SS_DOWN_SCALE * y);
			if (depth > 0)
			{
				// apply block average
				TDepth sum = 0;
				TCount count = 0;
				for (unsigned int x_ = KF_SS_DOWN_SCALE * x; x_ < KF_SS_DOWN_SCALE * (x + 1); x_++)
				{
					for (unsigned int y_ = KF_SS_DOWN_SCALE * y; y_ < KF_SS_DOWN_SCALE * (y + 1); y_++)
					{
						// check depth boundary
						TDepth depth_ = depthMap_(x_, y_);
						if (std::abs(depth - depth_) < KF_SS_SIGMA_DEPTH)
						{
							sum += depth_;
							count++;
						}
					}
				}

				// set average depth
				depthMap(x, y) = sum / count;
			}
			else
			{
				// set invalid depth
				depthMap(x, y) = 0;
			}
		}
	}

	void CFrame::applySubSample() const
	{
		// apply sub-sampling for each image pair
		for (unsigned int l = 1; l < m_pyramid.size(); l++)
		{
			// get lower level image and upper level image
			const CImage& image_ = m_pyramid.at(l - 1);
			const CImage& image = m_pyramid.at(l);

			// apply sub-sampling (CUDA)
			kernelSubSample<<<image.depthMap.dimGrid, image.depthMap.dimBlock>>>(
				image_.depthMap, image.depthMap,
				image.width, image.height);
			if (cudaDeviceSynchronize() != cudaSuccess)
			{
				throw std::runtime_error("kf::CFrame::applySubSample - failed to apply sub-subsampling");
			}

			// apply sub-sampling (CUDA w/ bilateral filter)
			kernelSubSample<<<image.depthMapF.dimGrid, image.depthMapF.dimBlock>>>(
				image_.depthMapF, image.depthMapF,
				image.width, image.height);
			if (cudaDeviceSynchronize() != cudaSuccess)
			{
				throw std::runtime_error("kf::CFrame::applySubSample - failed to apply sub-subsampling (w/ bilateral filter)");
			}

#if KF_DEBUG
			// DEBUG: download depth map
			image.depthMap.download();

			// DEBUG: download depth map (w/ bilateral filter)
			image.depthMapF.download();
#endif
		}
	}

	__global__ void kernelClearMask(
		TMaskMap maskMap,
		unsigned int width, unsigned int height)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			// clear mask data
			// NOTE: clear mask data to True
			maskMap(x, y) = true;
		}
	}

	void CFrame::clearMask() const
	{
		// clear mask (CUDA)
		kernelClearMask<<<m_maskMap.dimGrid, m_maskMap.dimBlock>>>(
			m_maskMap,
			m_uiWidth, m_uiHeight);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CFrame::clearMask - failed to clear mask");
		}
	}
}
