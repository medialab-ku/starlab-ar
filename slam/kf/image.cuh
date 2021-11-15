#pragma once

#include "setting.cuh"
#include "type.cuh"

#include <cmath>
#include <algorithm>

namespace kf
{
	class CImage
	{
	 private:
		unsigned int m_uiWidth, m_uiHeight;
		SIntrinsic m_intrinsic;

		TDepthMap m_depthMap, m_depthMapF;
		TVertexMap m_vertexMap, m_vertexMapF;
		TNormalMap m_normalMap, m_normalMapF;
		TValidityMap m_validityMap;

	 public:
		const unsigned int& width = m_uiWidth;
		const unsigned int& height = m_uiHeight;
		const SIntrinsic& intrinsic = m_intrinsic;

		const TDepthMap& depthMap = m_depthMap;
		const TDepthMap& depthMapF = m_depthMapF;
		const TVertexMap& vertexMap = m_vertexMap;
		const TVertexMap& vertexMapF = m_vertexMapF;
		const TNormalMap& normalMap = m_normalMap;
		const TNormalMap& normalMapF = m_normalMapF;
		const TValidityMap& validityMap = m_validityMap;

	 public:
		CImage() = delete;
		CImage(
			unsigned int uiWidth, unsigned int uiHeight,
			SIntrinsic intrinsic);
		~CImage();

	 public:
		void input(const TDepth* pDepth) const;
		void filter() const;
		void compute() const;

	 private:
		void applyBilateralFilter() const;
		void computeVertexMap() const;
		void computeNormalMap() const;
		void computeValidityMap() const;
	};
}
