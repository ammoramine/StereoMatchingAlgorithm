#ifndef CENSUS_INCLUDED
#define CENSUS_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <opencv/highgui.h>
#include <iostream>
#include <vector>
#include "ROF3D.h"
#include "someTools.h"

class Census {
	public:
		// void computeNeighbors(const cv::Vec<int,2> &curPixel,std::vector< cv::Vec<int,2> > &neighbors);
		// void getIntensityNeighbors(const cv::Vec<int,2> &curPixel,double intensityNeighbors[],const double * Im1,const double * I,const double * Ip1);
		// void ternaryCensusSignature(const cv::Mat &image,cv::Mat &ternaryCensusSignature);
		Census(const cv::Mat &image1,const cv::Mat &image2,DataTerm &data_term);
		double hammingDistance(const cv::Mat &ternarySignature1,const cv::Mat &ternarySignature2);
		void computeCensusSignature(const cv::Mat &image,cv::Mat &ternaryCensusSignature);
		void computeDataTerm();
		void computeDataTermSubPixel();
		// void showImages(const std::vector<cv::Mat> &images);
		// void data_term_census(const cv::Mat &image1,const cv::Mat &image2,cv::Mat &g,double epsilon=0.5);
	private:
		cv::Mat  m_image1;
		cv::Mat  m_image2;
		cv::Mat  m_CensusSignature_1;
		cv::Mat  m_CensusSignature_2;
		cv::Mat m_dataterm;
		// DataTerm m_dataTerm;
		double m_offset;
		double m_stepDisparity;
		// int m_disparityExtent;


};
#endif