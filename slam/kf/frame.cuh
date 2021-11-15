#pragma once

#include "image.cuh"
#include "timer.cuh"

#include <vector>

namespace kf
{
	typedef std::vector<CImage> TPyramid;

	class CFrame
	{
	 private:
		unsigned int m_uiWidth, m_uiHeight, m_uiLevel;
		SIntrinsic m_intrinsic;
		TPose m_pose;

		TPyramid m_pyramid;
		TMaskMap m_maskMap;

	 public:
		const unsigned int& width = m_uiWidth;
		const unsigned int& height = m_uiHeight;
		const unsigned int& level = m_uiLevel;
		const SIntrinsic& intrinsic = m_intrinsic;
		TPose& pose = m_pose;

		const TPyramid& pyramid = m_pyramid;
		const TMaskMap& maskMap = m_maskMap;

	 public:
		CFrame() = delete;
		CFrame(
			unsigned int uiWidth, unsigned int uiHeight, unsigned int uiLevel,
			SIntrinsic intrinsic, TPose pose = TPose::Identity());
		~CFrame();

	 public:
		void preprocess(const void* pDepth) const;
		void mask(const void* pMask) const;
		void clear() const;

	 private:
		void applySubSample() const;
		void clearMask() const;
	};
}
