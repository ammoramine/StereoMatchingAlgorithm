#ifndef SOMETOOLS_H_INCLUDED
#define SOMETOOLS_H_INCLUDED
#include <opencv/highgui.h>
#include <opencv/cv.h>
// #include <opencv2/core/core.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
// #include <opencv2/imgproc/imgproc.hpp>
#include <stdexcept>
#include <time.h>
#include <fstream>
#include <math.h>
extern "C"
{
#include "iio.h"
#include "zoom.h"
}
// #include </home/amine/opencv_install/opencv/modules/core/include/opencv2/core/types.hpp>

cv::Mat getLayer(const cv::Mat Matrix3D,int layer_number);
void getLayer3D(const cv::Mat &matrix3D,int layer_number,cv::Mat &layer);
void getLayer2D(const cv::Mat &matrix2D,int layer_number,cv::Mat &layer1D);
void getLayer3DBeta(const cv::Mat &matrix3D,int layer_number,cv::Mat &layer1);
void cast3DMatrixTo2DMatrix(const cv::Mat &matrix3D, cv::Mat &matrix2D);

cv::Mat getLayer2DOld(const cv::Mat &matrix2D,int layer_number);

cv::Mat getRow4D(const cv::Mat &Matrix4D,int numberRow,bool newOne=false);
void getRow4D(const cv::Mat &Matrix4D,int numberRow,cv::Mat &extractedMatrix);
void getRow3D(const cv::Mat &Matrix3D,int numberRow,cv::Mat &extractedMatrix);
void getRow2D(const cv::Mat &Matrix2D,int numberRow,cv::Mat &extractedMatrix);

void setRow3D(const cv::Mat &Matrix3Di,int i,cv::Mat &Matrix3D);

void printContentsOf3DCVMat(const cv::Mat &matrix,bool writeOnFile=true,std::string filename="FileStorage.txt");
void castCVMatTovector_double(const cv::Mat &matrix,std::vector<double> &vector);
void testLayer3D();
void testLayer2D();
void testLayer2DBis();

void writeImageOnFloat(const cv::Mat &image,const std::string &name);
void resizeWithShannonInterpolation( cv::Mat &image,cv::Mat &resizedImage,int zoom=2);



#endif