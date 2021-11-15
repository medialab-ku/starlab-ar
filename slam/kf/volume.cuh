#pragma once

#include <stdexcept>

namespace kf
{
	template<typename T, unsigned int C = 1>
	class CVolume
	{
	 private:
		unsigned int m_uiWidth, m_uiHeight, m_uiDepth;
		T* h_pData, * d_pData;

		dim3 m_dimBlock, m_dimGrid;

	 public:
		const dim3& dimBlock = m_dimBlock;
		const dim3& dimGrid = m_dimGrid;

	 public:
		CVolume() = delete;

		CVolume(unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth) :
			m_uiWidth(uiWidth), m_uiHeight(uiHeight), m_uiDepth(uiDepth),
			h_pData(nullptr), d_pData(nullptr),
			m_dimBlock(KF_VOL_BLOCK_DIM_X, KF_VOL_BLOCK_DIM_Y, KF_VOL_BLOCK_DIM_Z),
			m_dimGrid(
				(m_uiWidth + KF_VOL_BLOCK_DIM_X - 1) / KF_VOL_BLOCK_DIM_X,
				(m_uiHeight + KF_VOL_BLOCK_DIM_Y - 1) / KF_VOL_BLOCK_DIM_Y,
				(m_uiDepth + KF_VOL_BLOCK_DIM_Z - 1) / KF_VOL_BLOCK_DIM_Z)
		{

		}

		unsigned int size() const
		{
			return m_uiWidth * m_uiHeight * m_uiDepth;
		}

		T* data() const
		{
			return h_pData;
		}

		void malloc()
		{
			if (h_pData || d_pData)
			{
				throw std::runtime_error("kf::CVolume::malloc - allocated memory");
			}

			h_pData = new T[C * size()];
			if (!h_pData)
			{
				throw std::runtime_error("kf::CVolume::malloc - failed to allocate Host memory");
			}

			if (cudaMalloc(&d_pData, sizeof(T) * C * size()) != cudaSuccess)
			{
				throw std::runtime_error("kf::CVolume::malloc - failed to allocate Device memory");
			}
		}

		void free()
		{
			if (!h_pData || !d_pData)
			{
				throw std::runtime_error("kf::CVolume::free - unallocated memory");
			}

			delete[] h_pData;
			h_pData = nullptr;

			if (cudaFree(d_pData) != cudaSuccess)
			{
				throw std::runtime_error("kf::CVolume::free - failed to free Device memory");
			}
			d_pData = nullptr;
		}

		void upload(unsigned int uiSize = 0) const
		{
			if (uiSize == 0)
			{
				uiSize = size();
			}

			if (!h_pData || !d_pData)
			{
				throw std::runtime_error("kf::CVolume::upload - unallocated memory");
			}

			if (cudaMemcpy(d_pData, h_pData, sizeof(T) * C * uiSize, cudaMemcpyHostToDevice) != cudaSuccess)
			{
				throw std::runtime_error("kf::CVolume::upload - failed to copy data from Host to Device");
			}
		}

		void download(unsigned int uiSize = 0) const
		{
			if (uiSize == 0)
			{
				uiSize = size();
			}

			if (!h_pData || !d_pData)
			{
				throw std::runtime_error("kf::CVolume::download - unallocated memory");
			}

			if (cudaMemcpy(h_pData, d_pData, sizeof(T) * C * uiSize, cudaMemcpyDeviceToHost) != cudaSuccess)
			{
				throw std::runtime_error("kf::CVolume::download - failed to copy data from Device to Host");
			}
		}

		__device__ T& operator()(unsigned int uiX, unsigned int uiY, unsigned int uiZ, unsigned int uiC = 0) const
		{
			return d_pData[uiC + C * uiX + C * m_uiWidth * uiY + C * m_uiWidth * m_uiHeight * uiZ];
		}
	};
}
