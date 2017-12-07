#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/gpu/gpu.hpp>
// #include <opencv2/ocl/ocl.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include "census_computation.h"


//// the ternaryCensusSignature needs to be ordered for the of the computation of the census transform, we choose the following order:
// .....


void computeNeighbors(const cv::Vec<int,2> &curPixel,std::vector< cv::Vec<int,2> > &neighbors)
{
	int i=curPixel[0];int j=curPixel[1];
	neighbors[0]=cv::Vec<int,2>(i-1,j);
	neighbors[1]=cv::Vec<int,2>(i-1,j+1);
	neighbors[2]=cv::Vec<int,2>(i,j+1);
	neighbors[3]=cv::Vec<int,2>(i+1,j+1);
	neighbors[4]=cv::Vec<int,2>(i+1,j);
	neighbors[5]=cv::Vec<int,2>(i+1,j-1);
	neighbors[6]=cv::Vec<int,2>(i,j-1);
	neighbors[7]=cv::Vec<int,2>(i-1,j-1);

}
void getIntensityNeighbors(const cv::Vec<int,2> &curPixel,double intensityNeighbors[],const double * Im1,const double * I,const double * Ip1)
{
	//intNeighbors is the intensity of the neghbors of the pixel, we begin by the upper pixel and then the 7 others in a clockwise direction. Im1, I and Ip1 are respectively a double pointor on the array of the above the current row, a double pointor on the current row and a pointor on the row below the current one
	int j=curPixel[1];
	intensityNeighbors[0]=Im1[j];
	intensityNeighbors[1]=Im1[j+1];
	intensityNeighbors[2]=I[j+1];
	intensityNeighbors[3]=Ip1[j+1];
	intensityNeighbors[4]=Ip1[j];
	intensityNeighbors[5]=Ip1[j-1];
	intensityNeighbors[6]=I[j-1];
	intensityNeighbors[7]=Im1[j-1];

}
// double s(double I1,double I1,epsilon=0.01)
// {

// }
void ternaryCensusSignature(const cv::Mat &image,cv::Mat &ternaryCensusSignature,double epsilon)
{
	int size[]={image.size[0],image.size[1],8}; //8 corresponds to the dmimension of the beighborhood-1, here it's the square around the pixel
	cv::Vec<int,2> curPixel;
	std::vector<cv::Vec<int,2> > neighbors;neighbors.resize(8);

	ternaryCensusSignature=cv::Mat(3,size,CV_64FC1, 0.0);
	cv::Mat ternaryCensusSignaturei;
	// cv::Mat imageim1;
	// cv::Mat imagei; 
	// cv::Mat imageip1;
	cv::Mat ternaryCensusSignatureij;
	// cv::Mat imageij;
	double intensityNeighbors[8];
	for (int i=1;i<size[0]-1;i++)
	{
		ternaryCensusSignaturei=MatchingAlgorithm::getRow3D(ternaryCensusSignature,i);
		const double *imageim1=image.ptr<double>(i-1);
		const double *imagei=image.ptr<double>(i);
		const double *imageip1=image.ptr<double>(i+1);
		curPixel[0]=i;
		for (int j=1;j<size[1]-1;j++)
		{
			double * ternaryCensusSignatureij =ternaryCensusSignaturei.ptr<double>(j);
			curPixel[1]=j;
			// computeNeighbors(curPixel,neighbors);
			double Icur=imagei[j];
			getIntensityNeighbors(curPixel,intensityNeighbors,imageim1,imagei,imageip1);
			for (int k=0;k<size[2];k++)
			{
				// int iNgb=neighbors[k][0];int jNgb=neighbors[k][1];
				// if (iNgb=i-1)
				// double INgb=
				// if (I)
				double a=intensityNeighbors[k]-Icur;
				// if (a<-epsilon)
				// {
				// 	ternaryCensusSignatureij[k]=0.0;
				// }
				// else if(a>epsilon)
				// {
				// 	ternaryCensusSignatureij[k]=2.0;
				// }
				if (a<0)
				{
					ternaryCensusSignatureij[k]=0.0;
				}
				else
				{
					ternaryCensusSignatureij[k]=1.0;
				}
			}
		}
	}
}
double hammingDistance(double *p,double*q,int lengthArray)
{
	double result=0.0;
	for (int i=0;i<lengthArray;i++)
	{
		if(p[i]!=q[i])
		{
			result+=1;
		}
	}
	return result;
}
void data_term_census(const cv::Mat &image1,const cv::Mat &image2,cv::Mat &g,double epsilon)
{
	// we suppose that the cv::Mat object have been initialized before to the value  g=cv::Mat(3, size, CV_64FC1, 500.0); with size linked to the size of images: image1 and image2.

	 //the convention y,x,t
	// g=cv::Mat(3, size, CV_64FC1, 0.0);
	int size[3] = { g.size[0], g.size[1], g.size[2] };
	cv::Mat ternaryCensusSignature1;
	cv::Mat ternaryCensusSignature2;
	cv::Mat ternaryCensusSignature1i;
	cv::Mat ternaryCensusSignature2i;
	cv::Mat ternaryCensusSignature1ij;
	cv::Mat ternaryCensusSignature2ij;	
	ternaryCensusSignature(image1,ternaryCensusSignature1,epsilon);
	ternaryCensusSignature(image2,ternaryCensusSignature2,epsilon);

	printContentsOf3DCVMat(ternaryCensusSignature1,true,"ternaryCensusSignature1");
	printContentsOf3DCVMat(ternaryCensusSignature2,true,"ternaryCensusSignature2");

		for (int i=1;i<size[0]-1;i++)
		{
			// double * deltaxPtr=deltax.ptr<double>(0);
			const double * image1iPtr= image1.ptr<double>(i);
			const double * image2iPtr= image2.ptr<double>(i);
			ternaryCensusSignature1i=MatchingAlgorithm::getRow3D(ternaryCensusSignature1,i);
			ternaryCensusSignature2i=MatchingAlgorithm::getRow3D(ternaryCensusSignature2,i);
			cv::Mat gi=MatchingAlgorithm::getRow3D(g,i);
			for (int j=1;j<size[1]-1;j++)
			{
				double * ternaryCensusSignature1ij =ternaryCensusSignature1i.ptr<double>(j);
								
				double * gij=gi.ptr<double>(j);
				int maxk=std::min(j,size[2]-1);
				for(int k=0;k<=maxk;k++)
				{
					double * ternaryCensusSignature2ijmk =ternaryCensusSignature2i.ptr<double>(j-k);
			// for (int km
					gij[k]=hammingDistance(ternaryCensusSignature1ij,ternaryCensusSignature2ijmk,8);
				}
				// delete m_gij;
			}
		}
			// delete m_image1iPtr;
			// delete m_image2iPtr;
}
	// else if (m_da
// }

void showImages(const std::vector<cv::Mat> &images)
{
	// for
	// cv::namedWindow("Output Image1");
	// cv::namedWindow("Output Image2");
	// m_image1->convertTo(*m_image1, CV_8U);
	// m_image2->convertTo(*m_image2, CV_8U);

	for (int i=0;i<images.size();i++)
	{
		const int j = i;
		std::ostringstream s;
		s << j;
		const std::string i_as_string(s.str());
		cv::imshow("Output Image"+i_as_string, images[i]);
	}
	cv::waitKey(0);
}
// }
// void helloThere()
// {
// 	std::cout<<"hello there"<<std::endl;
// }