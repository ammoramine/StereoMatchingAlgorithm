#ifndef ROF3DMULTISCALE_INCLUDED
#define ROF3DMULTISCALE_INCLUDED
#include "ROF3D.h"


class ROF3DMultiscale : public ROF3D
{
	public:
		// ROF3DMultiscale(const std::vector<cv::Mat>& data_terms,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double offset,double ratioGap,double precision);
		// ROF3DMultiscale(const std::vector<DataTerm>& data_terms,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double offset,double ratioGap,double precision);
		// ROF3DMultiscale(std::vector<const DataTerm>& data_terms,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double offset,double ratioGap);
		ROF3DMultiscale(const std::vector<DataTerm>& data_terms,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double ratioGap);
		void computeScales();
		void changeScale(const cv::Mat &matrix3DOld,cv::Mat &matrix3DNew,int scale);
		void changeScale(const cv::Mat &matrix3DOld,cv::Mat &matrix3DNew,const cv::Mat &newDataTerm);
	private:
		std::vector<int> m_scales;
		std::vector<cv::Mat> m_data_terms;
};
#endif