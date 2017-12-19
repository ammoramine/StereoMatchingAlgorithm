#ifndef COMPUTEOCCLUSION_INCLUDED
#define COMPUTEOCCLUSION_INCLUDED
#include <opencv/highgui.h>
#include "someTools.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
extern "C"
{
#include "iio.h"
}
class Occlusion
{
	public:
		Occlusion(const std::string &disparityPath,const std::string &disparityReversePath,const std::string &disparityPathNoOcclusion);
		void computeCrossCheckingMask(cv::Mat &mask,const double &thresholdValue);
		void computeDisparityWithOcclusion();
		void writeDisparity();
	private:
		cv::Mat m_disparity;
		cv::Mat m_disparityReverse;
		cv::Mat m_maskNotOcclusion;// equals one when there no occlusion and 0 otherwise
		cv::Mat m_disparityNoOcclusion;
		std::string m_disparityNoOcclusionPath;

};


#endif
