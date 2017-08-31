#ifndef TOOLSFORREADING_H_INCLUDED
#define TOOLSFORREADING_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include <unistd.h>
static int verbose_flag;

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// extern char *optarg;
// extern int optind, opterr, optopt;
// extern const char** global_table_of_data_term;
// extern const std::vector<std::string> global_table_of_data_term;
extern const char * global_table_of_data_term[];

std::string compare_on_list(char * option,const char* listOfElements[],int sizeOflistOfElements,char * messageToPrint);
int get_dataterm_index(std::string name);
void read_option(int argc, char* argv[],cv::Mat &image1,cv::Mat &image2,std::string  &data_term_option,int &tsize,signed int &offset,int &Niter,std::string &path_to_disparity,std::string &method);


#endif