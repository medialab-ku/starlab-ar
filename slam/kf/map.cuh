#pragma once

#include <stdexcept>

namespace kf
{
	template<typename T, unsigned int C = 1>
	class CMap
	{
	 private:
		unsigned int m_uiWidth, m_uiHeight;
		T* h_pData, * d_pData;

		dim3 m_dimBlock, m_dimGrid;

	 public:
		const dim3& dimBlock = m_dimBlock;
		const dim3& dimGrid = m_dimGrid;

	 public:
		CMap() = delete;

		CMap(unsigned int uiWidth, unsigned int uiHeight) :
			m_uiWidth(uiWidth), m_uiHeight(uiHeight),
			h_pData(nullptr), d_pData(nullptr),
			m_dimBlock(KF_MAP_BLOCK_DIM_X, KF_MAP_BLOCK_DIM_Y),
			m_dimGrid(
				(m_uiWidth + KF_MAP_BLOCK_DIM_X - 1) / KF_MAP_BLOCK_DIM_X,
				(m_uiHeight + KF_MAP_BLOCK_DIM_Y - 1) / KF_MAP_BLOCK_DIM_Y)
		{

		}

		unsigned int size() const
		{
			return m_uiWidth * m_uiHeight;
		}

		T* data() const
		{
			return h_pData;
		}

		void malloc()
		{
			if (h_pData || d_pData)
			{
				throw std::runtime_error("kf::CMap::malloc - allocated memory");
			}

			h_pData = new T[C * size()];
			if (!h_pData)
			{
				throw std::runtime_error("kf::CMap::malloc - failed to allocate Host memory");
			}

			if (cudaMalloc(&d_pData, sizeof(T) * C * size()) != cudaSuccess)
			{
				throw std::runtime_error("kf::CMap::malloc - failed to allocate Device memory");
			}
		}

		void free()
		{
			if (!h_pData || !d_pData)
			{
				throw std::runtime_error("kf::CMap::free - unallocated memory");
			}

			delete[] h_pData;
			h_pData = nullptr;

			if (cudaFree(d_pData) != cudaSuccess)
			{
				throw std::runtime_error("kf::CMap::free - failed to free Device memory");
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
				throw std::runtime_error("kf::CMap::upload - unallocated memory");
			}

			if (cudaMemcpy(d_pData, h_pData, sizeof(T) * C * uiSize, cudaMemcpyHostToDevice) != cudaSuccess)
			{
				throw std::runtime_error("kf::CMap::upload - failed to copy data from Host to Device");
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
				throw std::runtime_error("kf::CMap::download - unallocated memory");
			}

			if (cudaMemcpy(h_pData, d_pData, sizeof(T) * C * uiSize, cudaMemcpyDeviceToHost) != cudaSuccess)
			{
				throw std::runtime_error("kf::CMap::download - failed to copy data from Device to Host");
			}
		}

		__device__ T& operator()(unsigned int uiX, unsigned int uiY, unsigned int uiC = 0) const
		{
			return d_pData[uiC + C * uiX + C * m_uiWidth * uiY];
		}
	};
}
