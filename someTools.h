#ifndef SOMETOOLS_H_INCLUDED
#define SOMETOOLS_H_INCLUDED
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdexcept>
#include <time.h>
#include <fstream>
#include <math.h>


cv::Mat getLayer(const cv::Mat Matrix3D,int layer_number);
cv::Mat getLayer3D(const cv::Mat &Matrix3D,int layer_number);
cv::Mat getLayer2D(const cv::Mat &Matrix2D,int layer_number);
cv::Mat getLayer2DOld(const cv::Mat &matrix2D,int layer_number);

cv::Mat getRow4D(const cv::Mat &Matrix4D,int numberRow,bool newOne=false);
cv::Mat getRow3D(const cv::Mat &Matrix3D,int numberRow);
cv::Mat getRow2D(const cv::Mat &Matrix2D,int numberRow);
void printContentsOf3DCVMat(const cv::Mat &matrix,bool writeOnFile=true,std::string filename="FileStorage.txt");
void castCVMatTovector_double(const cv::Mat &matrix,std::vector<double> &vector);
void testLayer3D();
void testLayer2D();
// void castCVMatTovector(const cv::Mat &matrix,std::vector<typeMatrix> &vector);

// template<typename typeForCasting>
// void castCVMatTovector(const cv::Mat &matrix,std::vector<typeForCasting> &vector)
// {
// 	// obtain iterator at initial position
// 	vector.resize(matrix.size[0]);
// 	cv::MatConstIterator_<typeForCasting> itMat=matrix.begin<typeForCasting>();
// 	std::vector<typeForCasting>::iterator itVec=vector.begin();
// 	// obtain end position
// 	cv::MatConstIterator_<typeForCasting> itMatend=matrix.end<typeForCasting>();
// 	for ( ; itMat!= itMatend; ++itMat) {
// 		(*itVec)=(*itMat);itVec+=1;
// 	}
// }


#endif