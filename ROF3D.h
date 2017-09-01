#ifndef ROF3D_INCLUDED
#define ROF3D_INCLUDED
#include <stdlib.h>
#include "ROF.h"
#include "MatchingAlgorithm.h"
#include "someTools.h"
#include <string>
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
#include "census_computation.h"
class ROF3D
{
	public:
		ROF3D(const cv::Mat & data_term,int m_Niter=100,const std::string &path_to_disparity="disparity.tif");
		void initf(double delta=1000);

		void launch();
		void iterate_algorithm();
		// void computeDual();
		void computeMinSumTV();
		void computeDisparity();

		cv::Mat getSolution();
		void testLab();


		void proxTVlStar(const cv::Mat &input,cv::Mat &output);
		void proxtauTVhStar(const cv::Mat &input,cv::Mat &output);
		void proxtauTVvStar(const cv::Mat &input,cv::Mat &output);


		void proxTVl(const cv::Mat &input,cv::Mat &output);
		void proxTVhOnTau(const cv::Mat &input,cv::Mat &output);
		void proxTVvOnTau(const cv::Mat &input,cv::Mat &output);

		// void step();
		double computeCostPrimal(const cv::Mat argument);
		double computeCostDual(const cv::Mat &x1,const cv::Mat &x2,const cv::Mat &x3);

		double computeTVHStar(const cv::Mat & argument,const double &precision=0.01);
		double computeTVVStar(const cv::Mat & argument,const double &precision=0.01);
		double computeTVLStar(const cv::Mat & argument,const double &precision=0.01);

		static double computeTV1DStar(const cv::Mat & argument,const double & precision=0.01);
		double computeTV1DStarWeighted(const cv::Mat & argument,const cv::Mat weight,const double & precision=0.01);


		double computeCostForArgumentTVl(const cv::Mat &l,const cv::Mat &argument);
		double computeCostForArgumentTVv(const cv::Mat &l,const cv::Mat &argument);
		double computeCostForArgumentTVh(const cv::Mat &l,const cv::Mat &argument);

		void testMinimialityOfSolutionTVL(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin);
		void testMinimialityOfSolutionTVV(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin);
		void testMinimialityOfSolutionTVH(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin);
		void testMinimalityOfSolution(int numberOfTests,double margin);
		void testContraintOnSolution(const cv::Mat &argminToTest);
		cv::Mat getSolutionOfOriginalProblem();


	private:
		cv::Mat m_g;
		int m_x_size;
		int m_y_size;
		int m_t_size;
		cv::Mat m_f;

		cv::Mat m_x1Current;cv::Mat m_x1Previous;
		cv::Mat m_x2Current;cv::Mat m_x2Previous;

		cv::Mat m_x1Bar;cv::Mat m_x2Bar;
		
		cv::Mat m_x3Current;//cv::Mat m_x3Previous;
		
		cv::Mat m_v;// the solution of the 3D ROF problem
		cv::Mat m_u;// the solution of he ishikawa formulation
		cv::Mat m_disparity;

		double m_tau;
		double m_t_current;
		double m_lambda;

		int m_iteration;
		int m_Niter;

		std::string m_path_to_disparity;
};

#endif