//
// Created by miru3137 on 2/17/21.
//

#include "image.cuh"

namespace kf
{
	CImage::CImage(
		unsigned int uiWidth, unsigned int uiHeight,
		SIntrinsic intrinsic) :
		m_uiWidth(uiWidth), m_uiHeight(uiHeight),
		m_intrinsic(intrinsic),
		m_depthMap(uiWidth, uiHeight), m_depthMapF(uiWidth, uiHeight),
		m_vertexMap(uiWidth, uiHeight), m_vertexMapF(uiWidth, uiHeight),
		m_normalMap(uiWidth, uiHeight), m_normalMapF(uiWidth, uiHeight),
		m_validityMap(uiWidth, uiHeight)
	{
		// allocate map memory
		m_depthMap.malloc();
		m_depthMapF.malloc();
		m_vertexMap.malloc();
		m_vertexMapF.malloc();
		m_normalMap.malloc();
		m_normalMapF.malloc();
		m_validityMap.malloc();
	}

	CImage::~CImage()
	{
		// free map memory
		m_depthMap.free();
		m_depthMapF.free();
		m_vertexMap.free();
		m_vertexMapF.free();
		m_normalMap.free();
		m_normalMapF.free();
		m_validityMap.free();
	}

	void CImage::input(const TDepth* pDepth) const
	{
		// check input depth data
		if (!pDepth)
		{
			throw std::runtime_error("kf::CImage::input - no depth data input");
		}

		// copy & upload depth data
		std::copy_n(pDepth, m_depthMap.size(), m_depthMap.data());
		m_depthMap.upload();
	}

	void CImage::filter() const
	{
		// apply bilateral filter
		applyBilateralFilter();
	}

	void CImage::compute() const
	{
		// compute vertex map, normal map, validity map
		computeVertexMap();
		computeNormalMap();
		computeValidityMap();
	}

	__global__ void kernelBilateralFilter(
		TDepthMap depthMap, TDepthMap depthMapF,
		unsigned int width, unsigned int height,
		int windowSize, float sigmaDepth, float sigmaSpace)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			TDepth d = depthMap(x, y);
			if (d > 0)
			{
				// for each pixel in window
				float sum = 0, norm = 0;
				for (int i = -windowSize; i <= windowSize; i++)
				{
					for (int j = -windowSize; j <= windowSize; j++)
					{
						int x_ = static_cast<int>(x) + i;
						int y_ = static_cast<int>(y) + j;
						if (x_ >= 0 && x_ < width && y_ >= 0 && y_ < height)
						{
							TDepth d_ = depthMap(x_, y_);
							if (d_ > 0)
							{
								// compute square terms
								float sqDiffDepth = (d - d_) * (d - d_);
								float sqDiffSpace = (x - x_) * (x - x_) + (y - y_) * (y - y_);
								float sqSigmaDepth = sigmaDepth * sigmaDepth;
								float sqSigmaSpace = sigmaSpace * sigmaSpace;

								// compute gaussian terms
								float gaussDepth = std::exp(-sqDiffDepth / (2 * sqSigmaDepth));
								float gaussSpace = std::exp(-sqDiffSpace / (2 * sqSigmaSpace));

								// sum filtering values
								sum += d_ * gaussDepth * gaussSpace;
								norm += gaussDepth * gaussSpace;
							}
						}
					}
				}

				// set filtering result
				depthMapF(x, y) = sum / norm;
			}
			else
			{
				// set invalid depth
				depthMapF(x, y) = 0;
			}
		}
	}

	void CImage::applyBilateralFilter() const
	{
		// apply bilateral filter (CUDA)
		kernelBilateralFilter<<<m_depthMapF.dimGrid, m_depthMapF.dimBlock>>>(
			m_depthMap, m_depthMapF,
			m_uiWidth, m_uiHeight,
			KF_BF_WINDOW_SIZE, KF_BF_SIGMA_DEPTH, KF_BF_SIGMA_SPACE);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CImage::applyBilateralFilter - failed to apply bilateral filtering");
		}

#if KF_DEBUG
		// DEBUG: download depth map (w/ bilateral filter)
		m_depthMapF.download();
#endif
	}

	__global__ void kernelVertexMap(
		TDepthMap depthMap, TVertexMap vertexMap,
		unsigned int width, unsigned int height,
		SIntrinsic intrinsic)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			TDepth& depth = depthMap(x, y);
			TVertex& vertex = vertexMap(x, y);
			if (depth > 0)
			{
				// compute vertex (inverse projection)
				vertex.x() = depth * ((float)x - intrinsic.cx) / intrinsic.fx;
				vertex.y() = depth * ((float)y - intrinsic.cy) / intrinsic.fy;
				vertex.z() = depth;
			}
			else
			{
				// set invalid vertex
				vertex.setZero();
			}
		}
	}

	void CImage::computeVertexMap() const
	{
		// compute vertex map (CUDA)
		kernelVertexMap<<<m_vertexMap.dimGrid, m_vertexMap.dimBlock>>>(
			m_depthMap, m_vertexMap,
			m_uiWidth, m_uiHeight,
			m_intrinsic);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CImage::computeVertexMap - failed to compute vertex map");
		}

		// compute vertex map (CUDA w/ bilateral filter)
		kernelVertexMap<<<m_vertexMapF.dimGrid, m_vertexMapF.dimBlock>>>(
			m_depthMapF, m_vertexMapF,
			m_uiWidth, m_uiHeight,
			m_intrinsic);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CImage::computeVertexMap - failed to compute vertex map (w/ bilateral filter)");
		}

#if KF_DEBUG
		// DEBUG: download vertex map
		m_vertexMap.download();

		// DEBUG: download vertex map (w/ bilateral filter)
		m_vertexMapF.download();
#endif
	}

	__global__ void kernelNormalMap(
		TDepthMap depthMap, TVertexMap vertexMap, TNormalMap normalMap,
		unsigned int width, unsigned int height)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x > 0 && x < width && y > 0 && y < height)
		{
			if (depthMap(x, y) > 0 && depthMap(x - 1, y) > 0 && depthMap(x, y - 1) > 0)
			{
				TVector dx = vertexMap(x, y) - vertexMap(x - 1, y);
				TVector dy = vertexMap(x, y) - vertexMap(x, y - 1);

				// compute normal vector
				normalMap(x, y) = dx.cross(dy).normalized();
			}
			else
			{
				// set invalid normal vector
				normalMap(x, y).setZero();
			}
		}
	}

	void CImage::computeNormalMap() const
	{
		// compute normal map (CUDA)
		kernelNormalMap<<<m_normalMap.dimGrid, m_normalMap.dimBlock>>>(
			m_depthMap, m_vertexMap, m_normalMap,
			m_uiWidth, m_uiHeight);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CImage::computeNormalMap - failed to compute normal map");
		}

		// compute normal map (CUDA w/ bilateral filter)
		kernelNormalMap<<<m_normalMapF.dimGrid, m_normalMapF.dimBlock>>>(
			m_depthMapF, m_vertexMapF, m_normalMapF,
			m_uiWidth, m_uiHeight);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CImage::computeNormalMap - failed to compute normal map (w/ bilateral filter)");
		}

#if KF_DEBUG
		// DEBUG: download normal map
		m_normalMap.download();

		// DEBUG: download normal map (w/ bilateral filter)
		m_normalMapF.download();
#endif
	}

	__global__ void kernelValidityMap(
		TNormalMap normalMap, TValidityMap validityMap,
		unsigned int width, unsigned int height)
	{
		unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < width && y < height)
		{
			// compute validity from normal vector
			validityMap(x, y) = !normalMap(x, y).isZero();
		}
	}

	void CImage::computeValidityMap() const
	{
		// compute validity map
		kernelValidityMap<<<m_validityMap.dimGrid, m_validityMap.dimBlock>>>(
			m_normalMap, m_validityMap,
			m_uiWidth, m_uiHeight);
		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("kf::CImage::computeValidityMap - failed to compute validity map");
		}

#if KF_DEBUG
		// DEBUG: download validity map
		m_validityMap.download();
#endif
	}
}
