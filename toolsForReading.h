#ifndef TOOLSFORREADING_H_INCLUDED
#define TOOLSFORREADING_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include <unistd.h>
#include "someTools.h"
extern "C"
{
#include "iio.h"
}
using namespace std;
static int verbose_flag;

// this module is used to read options given to the executable, it uses the  getopt and getopt_long functions that automate some of the chore involved in parsing typical unix command line options. 
extern const char * global_table_of_data_term[];

std::string compare_on_list(char * option,const char* listOfElements[],int sizeOflistOfElements,char * messageToPrint);
int get_dataterm_index(std::string name);
void read_option(int argc, char* argv[],cv::Mat &image1,cv::Mat &image2,std::string  &data_term_option,int &tsize,double &offset,double &ratioGap,int &Niter,double &zoom,std::string &path_to_disparity,std::string &path_to_initial_disparity,int &nbmaxThreadPoolThreading,std::string &method,bool &multiScale);
void readAndConvertImageToGray(const std::string &pathToImage,cv::Mat &output);

#endif