#pragma once

#include "map.cuh"
#include "volume.cuh"

#include <Eigen/Dense>

namespace kf
{
	struct SIntrinsic
	{
		float fx, fy;
		float cx, cy;
	};

	typedef float TDepth;
	typedef Eigen::Vector3f TVertex;
	typedef Eigen::Vector3f TNormal;
	typedef bool TValidity;
	typedef int TInstance;
	typedef bool TMask;
	typedef Eigen::Matrix4f TPose;
	typedef Eigen::Vector3f TVector;

	typedef CMap<TDepth> TDepthMap;
	typedef CMap<TVertex> TVertexMap;
	typedef CMap<TNormal> TNormalMap;
	typedef CMap<TValidity> TValidityMap;
	typedef CMap<TInstance> TInstanceMap;
	typedef CMap<TMask> TMaskMap;

	typedef Eigen::Vector3f TPosition;
	typedef float TTsdf;
	typedef float TWeight;
	typedef unsigned int TCount;
	typedef float TBinomial;

	typedef CVolume<TPosition> TPositionVolume;
	typedef CVolume<TTsdf> TTsdfVolume;
	typedef CVolume<TWeight> TWeightVolume;
	typedef CVolume<TCount> TCountVolumeA;
	typedef CVolume<TCount> TCountVolumeB;
	typedef CVolume<TBinomial> TBinomialVolume;
	typedef CVolume<TVertex, 5 * 3> TPolygonVolume;

	typedef Eigen::Matrix<float, 6, 6> TCholeskyA;
	typedef Eigen::Matrix<float, 6, 1> TCholeskyB;
	typedef float TError;
	typedef Eigen::Matrix<float, 6, 1> TJacobian;
	typedef float TResidual;
	typedef Eigen::Matrix<float, 6, 1> TCholeskyX;

	typedef CMap<TCholeskyA> TCholeskyMapA;
	typedef CMap<TCholeskyB> TCholeskyMapB;
	typedef CMap<TError> TErrorMap;
	typedef CMap<TCount> TCountMap;
}
