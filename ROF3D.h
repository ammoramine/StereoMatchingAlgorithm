#ifndef ROF3D_INCLUDED
#define ROF3D_INCLUDED
#include <stdlib.h>
#include "ROF.h"
// #include "MatchingAlgorithm.h"
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
#include <thread>
#include "threadpool.h"
#include <math.h>
#include <opencv2/opencv.hpp>
// #include "ROF3DMultiscale.h" 

// #include "iio.h"

extern "C"
{
#include "iio.h"
}
// a more compact structure to represent the dataTerm
// The 3D data term data_term is the main input for the ROF3D algorithm, that computes the solution of the Rudin–Osher–Fatem problem: 
	//argmin_{v}( Sigma g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+Sigma |v(i,j+1,k)-v(i,j,k)|+Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma |v(i,j,k)-m_f(i,j,k)|^2) with v(i,j,k)
//(with g=data_term)
	//from the term v, we  the disparity map is computed.

	// the term data_term(i,j,k) represents the cost  for matching the pixel (i,j) of the image 1 to the pixel (i,j+k*stepDisparity+offset) of the image 2, the offset being the smallest term of the interval of disparity that could be negative

	//These information on the disparity, are not be necessary to compute the solution of the ROF problem, but for the purpose of debugging, the disparity is computed each step of the algorithm.
	//In order to keep these informations about the disparity, and efficiently compute it inside this class, this structure is created, containing, the dataTerm (a 3D matrix) called matrix, the offset and the step as doubles. 
	// the step should be an inverse of integer. This term should be computed in the MatchingAlgorithm.h
struct DataTerm
{
	cv::Mat matrix;
	double offset;
	double stepDisparity;
};
class ROF3D
{
	public:
		// the constructor of ROF3D, compute the solution of the 3D ROF problem: 
		//argmin_{v}( Sigma g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+Sigma |v(i,j+1,k)-v(i,j,k)|+Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma |v(i,j,k)-m_f(i,j,k)|^2) with v(i,j,k)
		// the data_term should be
		// the disparity is deduced from the solution of the 3D ROF problem, by adding an offset
		ROF3D(const cv::Mat & data_term,int m_Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double offset,double ratioGap,const double &stepDisparity=1.0,double precision=0.0000001);
		ROF3D(const cv::Mat & data_term,int Niter,const std::string &path_to_disparity,size_t nbMaxThreads,double offset,double ratioGap,const cv::Mat &x1Current,const cv::Mat &x2Current,const cv::Mat &x3Current,const double &stepDisparity,double precision=0.0000001);

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


// parallelisable block for the computation of the proximal operators
		
		void proxTVLij(const cv::Mat &inputi,cv::Mat &outputi,const cv::Mat &gi,int j);
		void proxTVvOnTaupjk(const cv::Mat &inputppk, cv::Mat &output,int j,int k);//outputppk won't be used, but rather because the methods getLayer3D() and getLayer2D() return just a copy
		void proxTVhOnTauipk(const cv::Mat &inputi, cv::Mat &outputi,int k);

		void proxTVLijExtern(const cv::Mat &inputi,cv::Mat &outputi,const cv::Mat &gi);
		void proxTVvOnTaupjkExtern(const cv::Mat &inputppk, cv::Mat &output,int k);
		void proxTVhOnTauipkExtern(const cv::Mat &inputi, cv::Mat &outputi);

		void proxTVLijExternMultiple(const cv::Mat &input,cv::Mat &output,int beginIncluded,int endExcluded);
		void proxTVvOnTaupjkExternMultiple(const cv::Mat &input,cv::Mat &output,int beginIncluded,int endExcluded);
		void proxTVhOnTauipkExternMultiple(const cv::Mat &input,cv::Mat &output,int beginIncluded,int endExcluded);
		// void step();

		double computeCostPrimal(const cv::Mat &argument);
		double computeCostDual(const cv::Mat &x1,const cv::Mat &x2,const cv::Mat &x3);
		// double computeGapInfBorn(const cv::Mat &x1,const cv::Mat &x2,const cv::Mat &x3,const cv::Mat &primal);

		double computeTVHStar(const cv::Mat & argument);
		double computeTVVStar(const cv::Mat & argument);
		double computeTVLStar(const cv::Mat & argument);

		double computeTV1DStar(const cv::Mat & argument);
		double computeTV1DStarWeighted(const cv::Mat & argument,const cv::Mat weight);


		double computeCostForArgumentTVl(const cv::Mat &l,const cv::Mat &argument);
		double computeCostForArgumentTVv(const cv::Mat &l,const cv::Mat &argument);
		double computeCostForArgumentTVh(const cv::Mat &l,const cv::Mat &argument);

		void testMinimialityOfSolutionTVL(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin);
		void testMinimialityOfSolutionTVV(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin);
		void testMinimialityOfSolutionTVH(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin);
		void testMinimalityOfSolution(int numberOfTests,double margin);
		void testContraintOnSolution(const cv::Mat &argminToTest);
		cv::Mat getSolutionOfOriginalProblem();


	protected:
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
		double m_precision;// this parameter take into account the numerical inaccuracy for the computation of the conjuguate de la variation totale
		double m_offset;
		double m_ratioGap;// the ratio before stopping
		double m_CurrentRatioGap;// the ratio before stopping
		double m_intialDualPrimalGap;

		double m_stepDisparity;// linked with m_g, we should have that m_g/size[2]/m_stepDisparity= the interval of disparity
		size_t m_nbMaxThreads;

		int m_iteration;
		int m_Niter;

		std::string m_path_to_disparity;
		std::string m_path_to_initial_disparity;
};
// #include "ROF3DMultiscale.h"
#endif