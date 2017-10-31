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
cv::Mat getRow4D(const cv::Mat &Matrix4D,int numberRow,bool newOne=false);
cv::Mat getRow3D(const cv::Mat &Matrix3D,int numberRow);
cv::Mat getRow2D(const cv::Mat &Matrix2D,int numberRow);
void printContentsOf3DCVMat(const cv::Mat matrix,bool writeOnFile=true,std::string filename="FileStorage.txt");

void testLayer3D();
void testLayer2D();


#endif