#pragma once

#include "object.cuh"

namespace kf
{
	class CCamera
	{
	 private:
		unsigned int m_uiWidth, m_uiHeight;
		SIntrinsic m_intrinsic;
		TPose m_pose;

		TDepthMap m_depthMap;
		TVertexMap m_vertexMap;
		TNormalMap m_normalMap;
		TValidityMap m_validityMap;
		TInstanceMap m_instanceMap;

		TCholeskyMapA m_choleskyMapA;
		TCholeskyMapB m_choleskyMapB;
		TErrorMap m_errorMap;
		TCountMap m_countMap;

		float m_fTrackError;

	 public:
		const unsigned int& width = m_uiWidth;
		const unsigned int& height = m_uiHeight;
		const SIntrinsic& intrinsic = m_intrinsic;
		const TPose& pose = m_pose;

		const TDepthMap& depthMap = m_depthMap;
		const TVertexMap& vertexMap = m_vertexMap;
		const TNormalMap& normalMap = m_normalMap;
		const TValidityMap& validityMap = m_validityMap;
		const TInstanceMap& instanceMap = m_instanceMap;

		const float& trackError = m_fTrackError;

	 public:
		CCamera() = delete;
		CCamera(
			unsigned int uiWidth, unsigned int uiHeight,
			SIntrinsic intrinsic, TPose pose = TPose::Identity());
		~CCamera();

	 public:
		void raycast(const CObject& object) const;
		void track(CFrame& frame);
		void clear() const;

	 private:
		void raycastObject(const CObject& object) const;
		void trackFrame(CFrame& frame) const;
		void clearMap() const;
	};
}
