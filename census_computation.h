#ifndef CENSUS_COMPUTATION_INCLUDED
#define CENSUS_COMPUTATION_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include "census_computation.h"
#include "MatchingAlgorithm.h"


// void TernaryCensusSignature(cv::Mat image,)
// {

// }
void computeNeighbors(const cv::Vec<int,2> &curPixel,std::vector< cv::Vec<int,2> > &neighbors);
void getIntensityNeighbors(const cv::Vec<int,2> &curPixel,double intensityNeighbors[],const double * Im1,const double * I,const double * Ip1);
void ternaryCensusSignature(const cv::Mat &image,cv::Mat &ternaryCensusSignature);
double hammingDistance(double *p,double*q,int lengthArray);
void showImages(const std::vector<cv::Mat> &images);
void data_term_census(const cv::Mat &image1,const cv::Mat &image2,cv::Mat &g,double epsilon=0.5);

#endif