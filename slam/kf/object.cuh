#pragma once

#include "frame.cuh"

namespace kf
{

	class CObject
	{
	 private:
		static bool s_bInitMarchingCube;

		unsigned int m_uiWidth, m_uiHeight, m_uiDepth;
		TPosition m_center;
		float m_fVoxelSize;
		TInstance m_instance;

		TPositionVolume m_positionVolume;
		TTsdfVolume m_tsdfVolume;
		TWeightVolume m_weightVolume;

		TCountVolumeA m_countVolumeA;
		TCountVolumeB m_countVolumeB;
		TBinomialVolume m_binomialVolume;
		TPolygonVolume m_polygonVolume;

	 public:
		const unsigned int& width = m_uiWidth;
		const unsigned int& height = m_uiHeight;
		const unsigned int& depth = m_uiDepth;
		const TPosition& center = m_center;
		const float& voxelSize = m_fVoxelSize;
		const TInstance& instance = m_instance;

		const TPositionVolume& positionVolume = m_positionVolume;
		const TTsdfVolume& tsdfVolume = m_tsdfVolume;
		const TWeightVolume& weightVolume = m_weightVolume;
		const TBinomialVolume& binomialVolume = m_binomialVolume;
		const TPolygonVolume& polygonVolume = m_polygonVolume;

	 public:
		CObject() = delete;
		CObject(
			unsigned int uiWidth, unsigned int uiHeight, unsigned int uiDepth,
			float fCenterX, float fCenterY, float fCenterZ, float fVoxelSize,
			TInstance instance = 0);
		~CObject();

	 public:
		void integrate(const CFrame& frame) const;
		void applyMarchingCube() const;

	 private:
		void resetVolume() const;
		void updateVolume(const CFrame& frame) const;
	};
}
