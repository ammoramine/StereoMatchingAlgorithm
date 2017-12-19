#include "computeOcclusion.h"

Occlusion::Occlusion(const std::string &disparityPath,const std::string &disparityReversePath,const std::string &disparityNoOcclusionPath)
{
	m_disparity=cv::imread(disparityPath,cv::IMREAD_LOAD_GDAL); m_disparity.convertTo(m_disparity, CV_64FC1); 
 	m_disparityReverse=cv::imread(disparityReversePath,cv::IMREAD_LOAD_GDAL);m_disparityReverse.convertTo(m_disparityReverse, CV_64FC1);
	
	// m_disparity=disparity;
	// m_disparityReverse=disparityReverse;
	computeCrossCheckingMask(m_maskNotOcclusion,0.0);
	computeDisparityWithOcclusion();
	m_disparityNoOcclusionPath=disparityNoOcclusionPath;
	writeDisparity();
	// m_maskNotOcclusion=maskNotOcclusion;
}
void Occlusion::computeCrossCheckingMask(cv::Mat &mask,const double &thresholdValue)
{
	cv::Mat disparityi;cv::Mat disparityReversei;cv::Mat maski;
  	mask=cv::Mat(2, m_disparity.size,CV_64FC1, 0.0);

 	for (int i=0;i<mask.size[0];i++)
  	{
  		// getRow2D(image1,i,image1i);
  		getRow2D(m_disparity,i,disparityi);
  		getRow2D(m_disparityReverse,i,disparityReversei);
  		getRow2D(mask,i,maski);
  		for (int j=0;j<mask.size[1];j++)
  			{
  				int disparityij=int(floor(disparityi.at<double>(j)));
				// printContentsOf3DCVMat(disparityi,true,"disparityi");
  				int correspondingCol=j+disparityij;
  				maski.at<double>(j)=disparityi.at<double>(j)+disparityReversei.at<double>(correspondingCol);
  				// maski.at<double>(j)=-maski.at<double>(j);
  			}
  	}
    mask=cv::abs(mask);
  		// printContentsOf3DCVMat(mask,true,"before");
  		cv::threshold(mask,mask,thresholdValue,1.0,cv::THRESH_BINARY_INV);
  		double dNaN = std::numeric_limits<double>::quiet_NaN();
  		for (int i=0;i<mask.size[0];i++)
  		{
  			cv::Mat maski;getRow2D(mask,i,maski);
  			for (int j=0;j<mask.size[1];j++)
  				{
  					if (maski.at<double>(j)==0)
  					{
  						maski.at<double>(j)=dNaN;
  					}
  				}
  		}

  		printContentsOf3DCVMat(mask,true,"after");
}
void Occlusion::computeDisparityWithOcclusion()
{
	m_disparityNoOcclusion=m_disparity.mul(m_maskNotOcclusion);
}
void Occlusion::writeDisparity()
{
	m_disparityNoOcclusion.convertTo(m_disparityNoOcclusion,CV_32FC1);
	cv::Mat m_disparityNoOcclusionCopy=m_disparityNoOcclusion.clone();
	iio_write_image_float(strdup(m_disparityNoOcclusionPath.c_str()),(float *)m_disparityNoOcclusionCopy.data,m_disparityNoOcclusionCopy.size[1],m_disparityNoOcclusionCopy.size[0]);

}
