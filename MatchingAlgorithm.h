#ifndef MATCHING_ALGORITHM_H_INCLUDED
#define MATCHING_ALGORITHM_H_INCLUDED
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

class MatchingAlgorithm
{

	public:
		MatchingAlgorithm(const cv::Mat &image1,const cv::Mat &image2);
		~MatchingAlgorithm();
		
		static cv::Mat  projCh(const cv::Mat &v);
		static cv::Mat  projCh_effic(const cv::Mat &v);

		static cv::Mat  projKh(const cv::Mat &phi,const cv::Mat &g);
		static cv::Mat  projKh_effic(const cv::Mat &phi,const cv::Mat &g);
		
		static cv::Mat gradh(const cv::Mat &vect);
		static cv::Mat gradh_effic(const cv::Mat &vect);

		static cv::Mat divh(const cv::Mat &vect);
		static cv::Mat divh_effic(const cv::Mat &v);

		// void  vproj(const std::vector<cv::Mat> &v,std::vector<cv::Mat> vproj); //= projC(v)

		void printProperties();
		void helpDebug();
		void showImages();
		void data_term();
		void data_term_effic();
		void init();
		double computePrimalDualGap();

		void iterate_algorithm();
		void launch();
		void disparity_estimation();
		static cv::Mat getLayer(cv::Mat Matrix3D,int layer_number);
		static cv::Mat getRow(const cv::Mat &Matrix4D,int numberRow,bool newOne=false);
		static cv::Mat getRow3D(const cv::Mat &Matrix3D,int numberRow);
		static cv::Mat getRow2D(const cv::Mat &Matrix2D,int numberRow);
		static void printContentsOf3DCVMat(const cv::Mat matrix,bool writeOnFile=true,std::string filename="FileStorage.txt");


	private:

		cv::Mat *m_image1;
		cv::Mat *m_image2;
		cv::Mat m_g;
		cv::Mat m_phih;
		cv::Mat m_vbar;
		cv::Mat m_v;
		cv::Mat m_disparity;

		int m_x_size;
		int m_y_size;
		double m_mu;
		double m_tau;
		double m_sigma;

		int m_iteration;
		double m_gap;
		double m_gapInit;
		double m_factor;

		int m_Niter;
		int m_t_size;
		double m_s;
		// std::vector<std::vector<double> > m_cost;
		// std::deque<int> m_indexOrderedImages; //ordered index images that we wish to construct
		// std::deque<int> m_distances;//equivalent distances of the m_indexOrderedImages

};
#endif