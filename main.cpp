#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/gpu/gpu.hpp>
// #include <opencv2/ocl/ocl.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include "MatchingAlgorithm.h"
#include "toolsForReading.h"
#include "ROF.h"
#include "ROF3DMultiscale.h"
#include "someTools.h"
#include "computeOcclusion.h"
#include <fstream>
#include "computeOcclusion.h"
// #define IIO_DISABLE_LIBJPEG
extern "C"
{
#include "iio.h"
#include "zoom.h"
}
int main(int argc, char* argv[])
{
	// // testLayer3D()
	// double argumentArray[]={0.5,0.1,-0.5,-0.1,-1.0,1.011};
	// cv::Mat argument(1,6,CV_64FC1,argumentArray);
	// std::cout<<ROF3D::computeTV1DStar(argument)<<"\n"<<std::endl;

	// throw std::invalid_argument( "testing the algorithm" );
// std::cout<<"it's over"<<std::endl;
	bool dirtyTest=false;
	if (!dirtyTest)
	// if (not(1))
	{
	cv::Mat image1;//=cv::imread("input_pair/rectified_ref.tif");//, cv::IMREAD_LOAD_GDAL);
	cv::Mat image2;//=cv::imread("input_pair/rectified_sec.tif");//, cv::IMREAD_LOAD_GDAL);
	std::string data_term_option;
	// read_option(argc,argv,image1,image2,data_term_option);
	int t_size;double offset;int Niter;std::string path_to_disparity;int nbmaxThreadPoolThreading;double zoom;std::string method;std::string path_to_initial_disparity;double ratioGap;bool multiscale;
	read_option(argc,argv,image1,image2,data_term_option,t_size,offset,ratioGap,Niter,zoom,path_to_disparity,path_to_initial_disparity,nbmaxThreadPoolThreading,method,multiscale);
	// MatchingAlgorithm theAlgorithm = MatchingAlgorithm(image1,image2,data_term_option, t_size,offset,ratioGap,Niter,path_to_disparity,path_to_initial_disparity,nbmaxThreadPoolThreading,method);

	// int zoom=1;
	// cv::Mat image1_resized;cv::Mat image1Copy=image1.clone();
	// resizeWithShannonInterpolation(image1Copy,image1_resized,zoom);
	// cv::Mat image2_resized;cv::Mat image2Copy=image2.clone();
	// resizeWithShannonInterpolation(image2Copy,image2_resized,zoom);


	MatchingAlgorithm theAlgorithm = MatchingAlgorithm(image1,image2,data_term_option, t_size,offset,ratioGap,Niter,path_to_disparity,path_to_initial_disparity,zoom,nbmaxThreadPoolThreading,method,multiscale);
	

	int found=path_to_disparity.find_first_of(".");
	std::string path_to_disparity_reverse=path_to_disparity;
	path_to_disparity_reverse.insert(found,"_reverse");

	MatchingAlgorithm theAlgorithm_reverse = MatchingAlgorithm(image2,image1,data_term_option, t_size,-(offset+t_size),ratioGap,Niter,path_to_disparity_reverse,path_to_initial_disparity,zoom,nbmaxThreadPoolThreading,method,multiscale);

	int found1=path_to_disparity.find_first_of(".");
	std::string path_to_disparity_no_occlusion=path_to_disparity;
	path_to_disparity_no_occlusion.insert(found,"_noOcclusion");
	Occlusion occlusion=Occlusion(path_to_disparity,path_to_disparity_reverse,path_to_disparity_no_occlusion);
}
else
{//here is place from some dirty tests
	
 //  cv::Mat disparity=cv::imread("disparityGrayMars_census.tif",cv::IMREAD_LOAD_GDAL); disparity.convertTo(disparity, CV_64FC1); 
 //  cv::Mat disparityReverse=cv::imread("disparityGrayArt_census_reverse.tif",cv::IMREAD_LOAD_GDAL);disparityReverse.convertTo(disparityReverse, CV_64FC1);
	// Occlusion occlusion=Occlusion("disparityGrayMars_census.tif","disparityGrayMars_census_reverse.tif","disparityGrayMars_census_noOcclusion_beta.tif");
	// cv::Mat image=imread("image1_gray_Art.tif",cv::IMREAD_LOAD_GDAL);image.convertTo(image,CV_64FC1);
	// cv::Mat interpolatedImage;
	// Interpolation interpolation=Interpolation(image,2,interpolatedImage);

	// Occlusion occlusion=Occlusion("disparityGrayArt_absdiff.tif","disparityGrayArt_absdiff_reverse.tif",disparityNoOcclusion);
  // cv::Mat image1=cv::imread("image1_gray_Art.tif",cv::IMREAD_LOAD_GDAL);
  // cv::Mat image2=cv::imread("image2_gray_Art.tif",cv::IMREAD_LOAD_GDAL);

  // cv::Mat image1i;
  // cv::Mat disparityi;cv::Mat disparityReversei;cv::Mat maski;
  // cv:: Mat mask=cv::Mat(2, disparity.size,CV_64FC1, 0.0);
  // cv::Mat   maskBinary;

  // // printContentsOf3DCVMat(disparity,true,"disparity");
  // // printContentsOf3DCVMat(disparityReverse,true,"disparityReverse");


  // for (int i=0;i<mask.size[0];i++)
  // 	{
  // 		// getRow2D(image1,i,image1i);
  // 		getRow2D(disparity,i,disparityi);
  // 		getRow2D(disparityReverse,i,disparityReversei);
  // 		getRow2D(mask,i,maski);
  // 		for (int j=0;j<mask.size[1];j++)
  // 			{
  // 				int disparityij=int(floor(disparityi.at<double>(j)));
		// 		// printContentsOf3DCVMat(disparityi,true,"disparityi");
  // 				int correspondingCol=j+disparityij;
  // 				maski.at<double>(j)=disparityi.at<double>(j)+disparityReversei.at<double>(correspondingCol);
  // 				// maski.at<double>(j)=-maski.at<double>(j);
  // 			}
  // 	}
  // 		cv::threshold(mask,maskBinary,1.0,1.0,cv::THRESH_BINARY_INV);
  		
  // 		// printContentsOf3DCVMat(disparity,true,"disparity");

  // 		cv::Mat disparityCopy=disparity.mul(maskBinary);

  // 	  	// printContentsOf3DCVMat(disparityCopy,true,"disparityAfter");
  	  	
  // 	  	// printContentsOf3DCVMat(mask,true,"mask");
  // 	  	// printContentsOf3DCVMat(maskBinary,true,"maskBinary");
		
		// // cv::Mat disparityCopy=disparity.clone();
  //      	disparityCopy.convertTo(disparityCopy,CV_32FC1);
  //   	iio_write_image_float("disparityAfter.tif",(float *)disparityCopy.data,disparityCopy.size[1],disparityCopy.size[0]);
       	// printContentsOf3DCVMat(disparityCopy,true,"disparityCopy");

	// printContentsOf3DCVMat(disparityCopy,true,"disparityCopy");
	// bool continuity=disparityCopy.isContinuous();s
    // cv::imwrite(m_path_to_disparity,m_disparity);
  // iio_write_image_float("mask.tif",(float*)mask.data,mask.size[1],mask.size[0]);
  // disparity=disparity.mul(mask);

  // cv::Mat m_disparity=cv::Mat(outputTemp.size[0],outputTemp.size[1],CV_32FC1,10.0);
  // printContentsOf3DCVMat(m_disparity,true,"image2Content");
  // iio_write_image_float("image2.tif",(float*)m_disparity.data,m_disparity.size[1],m_disparity.size[0]);


  // int a=outputTemp.channels();
  // std::string b=outputTemp.depth();
  // std:cout<<outputTemp.depth()<<std::endl;
  // float r[100];
  // printContentsOf3DCVMat(outputTemp,true,"mgm_disp_neg");
  // boost::gil::gray32f_view_t dst=boost::gil::interleaved_view(outputTemp.size[0],outputTemp.size[1],(boost::gil::gray32f_pixel_t*)r,outputTemp.size[0]);
  // boost::gil::write_view("mgm_disp_neg_rewritten.tif",(any_image_view<boost::gil::gray32f_view_t>)dst);
  // std::string path=std::string("mgm_disp_neg_rewritten.tif");
  // boost::gil::tiff_write_view<boost::gil::gray32f_view_t>("mgm_disp_neg_rewritten.tif",dst);
  // cv::FileStorage fs("mgm_disp_neg_rewritten.tif", cv::FileStorage::WRITE );
  // fs << "Mat" << outputTemp;
  // cv::Mat outputTempNew;
  // outputTemp.convertTo(outputTempNew, CV_16FC1);
  // printContentsOf3DCVMat(outputTempNew,true,"mgm_disp_neg_after_conversion");
  // cv::imwrite("mgm_disp_neg_rewritten.tif",outputTempNew);
  // char * dispartityData=new char[463*370];//=disparity.ptr<char> 
  // ifstream myFile ("mgm_disp_neg.tif", ios::in | ios::binary);
  // myFile.read (dispartityData,1000000000000);
  // // cv::Mat mgmFile=
  // cv::Mat disparity=cv::Mat(463,370,CV_32S,dispartityData);
  // printContentsOf3DCVMat(disparity,true,"producedFile");
  // fstream myFile( "mgm_disp_census.tif", ios::in | ios::out | ios::binary );
  // cv::imwrite("rewritten.tif",mgmFile);
  // printContentsOf3DCVMat(mgmFile,true,"mgmFile");
}

		// (theAlgorithm.get_data_term()).copyTo(data_term);
	// }
	// ROF3D rof3D=ROF3D(data_term);
// if(0)
// {	
// 	int sizeCh[1]= {20};
// 	// int sizeCh[3]= {data_term.size[0],data_term.size[1],data_term.size[2]};
// 	cv::Mat M(1,sizeCh,CV_64FC1,1.0);
// 	cv::randu(M,1.0,15.5);
// 	std::vector<double> aVector;//(sizeCh[0],0.0);
// 	castCVMatTovector_double(M,aVector);
// 	printContentsOf3DCVMat(M,true,"M");	
// 	for (int i=0;i<aVector.size();i++) std::cout<<aVector[i]<<std::endl;
// 	// std::cout<<aVector<<std::endl;
// }
	// data_term.copyTo(M);
	// printContentsOf3DCVMat(M,true,"MB");
	// cv::randu(M,1.0,15.5);
	// printContentsOf3DCVMat(M,true,"M");
	// data_term=cv::Mat(3,sizeCh,CV_64FC1,1.0);
	// printContentsOf3DCVMat(data_term,true,"data_term_after");
	// ROF3D rof3D=ROF3D(M,10);
	// rof3D.testMinimalityOfSolution(10,0.00001);
	// rof3D.



	// double lArray[] = {-0.390742,0.296236,499.112,6.51223e-319,499.112,6.51223e-319,499.112,6.51223e-319};
	// double lArray[] = {1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1000};
	// double lArray[]= {500,0,500};
	// double lArray[] = {}
	// double lArray[] = {499.112,6.51223e-319,499.112,6.51223e-319};
	// double lArray[] = {500,0,500,0};

	// double lArray[] = {1,1,1,1};
	// double lArray[] = {1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1000};
  	// std::vector<double> l (lArray, lArray + sizeof(lArray) / sizeof(double) );
  	// double costArray[] = {6.93752,9.53447};
  	// double costArray[] = {1,1,1};
  	// double costArray[]={1,1,1,1};
  	// double costArray[] = {0.294118,0.294118,0.294118,0.294118,0,0.294118,0.294118,0.294118,0,0.294118,0,0,0.294118,0.294118,0,0,0.294118,0.294118,0,0.294118};
  	  	// double costArray[] = {1,1,1,1,1,1,1};
  	// double costArray[] = {0.294118,0.294118,0.294118,0.294118,0,0.294118,0.294118,0.294118,0,0.294118,0,0,0.294118,0.294118,0,0,0.294118,0.294118,0,0.294118};
  	// std::vector<double> cost (costArray, costArray + sizeof(costArray) / sizeof(double) );
  	// ROF rof=ROF(1.0,l,cost);
  	// rof.getSolution(true);
  	// rof.testMinimialityOfSolution(10000,0.0001);

// 2.60312 11.8222 5.23411 11.8126 1.52438 6.04725 12.6482 12.0096 3.68703 8.86533 6.33765 11.5027 11.3083 1.88262 9.43539 14.1303 6.69219 1.45258 14.2953 14.3229 5.15786 9.91403 3.66669 1.06507 12.4066 13.1654 13.7281 5.48689 7.29961 9.71131 9.44378 12.1289 1.4285 5.44143 11.1469 15.351 15.1396 6.6972 3.92183 12.3814










	// double m_sigma=0.45;
	// int size[2] = { 10, 10};
	// cv::Mat delta=cv::Mat(2, size, CV_64FC1, 10.0);
	// cv::Mat m_phih=cv::Mat(2, size, CV_64FC1, 7.0);
	// cv::Mat a=m_phih + m_sigma*delta;
	// double * aptr=a.ptr<double>(0);

// //////////////////////////////////////////////////////////////////////////////////////////////////// test de projCh et projCh_effic

// 			// std::cout<<"testing projCh "<<std::endl;
// 			// cv::Mat v;
// 				// cv::Mat vproj;
// 			int size1[3] = { m_y_size, m_x_size, m_t_size };
// 			cv::Mat v(3, size1, CV_64FC1, 1.0);
// 			srand (time(NULL));
// 			// for (int k=0;k<m_t_size;k++)
// 			// 	{
// 			// 		v[k]=cv::Mat::ones(m_y_size,m_x_size,CV_64FC1);
// 			// 		cv::randu(v[k],-3, 3);
// 			// 	}
// 			cv::randu(v,-3, 3);
// 			cv::Mat projv=projCh(v);
// 			cv::Mat projv_effic=projCh_effic(v);
// 			// vproj.resize(v.size());
// 				 // vproj(v,vproj);
// 			std::cout<<"\n print contents of vproj- before and after projection, choose a number between : "<<0<<" and tha maximum of dispartity map :"<<m_t_size<<"\n"<<std::endl;
// 			int number;
// 			std::cin>>number;
// 			// cv::Mat projv_layer=getLayer(projv,number);
// 			for (int i=0;i<m_y_size;i++)
// 				{
// 					//std::cout<<"\n"<<std::endl;
// 					for (int j=0;j<m_x_size;j++)
// 						{
// 								// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
// 							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
// 							std::cout<<"value on  pixel "<<i <<" and "<<j<<" of v :"<< v.at<double>(i,j,number)<<"  of vproj "<< projv.at<double>(i,j,number)<<" and of vproj_effic"<< projv_effic.at<double>(i,j,number)<<std::endl;
// 						}
// 				}
			

// //////////////////////////////////////////////////////////////////////////////////////////////////// 




//////////////////////////////////////////////////////////////////////////////////////////////////// test dde div_eff et grad_eff
	
	// int sizeCh[3]= {image1.size().height,image1.size().width,20};
	// int sizeKh[4]= {3,image1.size().height,image1.size().width,20};
	// cv::Mat matrixForGrad(3,sizeCh,CV_64FC1,0.0);cv::randu(matrixForGrad,0.0,1.0);
	// cv::Mat matrixForDiv(4,sizeKh,CV_64FC1,0.0);cv::randu(matrixForDiv,0.0,1.0);
	// cv::Mat gradh(4,sizeKh,CV_64FC1,0.0);

	// double durationStartdiv;
	// durationStartdiv = static_cast<double>(cv::getTickCount());
	// // clock_t tStartdiv = clock();
	// cv::Mat divv = MatchingAlgorithm::divh(matrixForDiv);
	// durationStartdiv = static_cast<double>(cv::getTickCount())-durationStartdiv;
	// durationStartdiv /= cv::getTickFrequency(); // the elapsed time in ms
	// // printf("Time taken for div: %.2fs\n", (double)(clock() - tStartdiv)/CLOCKS_PER_SEC);
	// printf("Time taken for div: %.2fs\n",durationStartdiv);

	// double durationStartdiv_effic;
	// durationStartdiv_effic = static_cast<double>(cv::getTickCount());
	// // clock_t tStartdivh_effic = clock();
	// cv::Mat divv1 = MatchingAlgorithm::divh_effic(matrixForDiv);
	// durationStartdiv_effic = static_cast<double>(cv::getTickCount())-durationStartdiv_effic;
	// durationStartdiv_effic /= cv::getTickFrequency(); // the elapsed time in ms
	// // printf("Time taken for divh_effic: %.2fs\n", (double)(clock() - tStartdivh_effic)/CLOCKS_PER_SEC);
	// printf("Time taken for div: %.2fs\n",durationStartdiv_effic);

	// cv::Mat resultdiv;cv::add(divv,-divv1,resultdiv);
	// std::cout<<cv::sum(divv)<<" and "<<cv::sum(divv1)<<std::endl;
	// std::cout<<cv::sum(resultdiv)<<std::endl;

	// clock_t tStartGrad = clock();
	// cv::Mat delta = MatchingAlgorithm::gradh(matrixForGrad) ;
	// printf("Time taken for grad: %.2fs\n", (double)(clock() - tStartGrad)/CLOCKS_PER_SEC);

	// clock_t tStartGradEffic = clock();
	// cv::Mat delta1 = MatchingAlgorithm::gradh_effic(matrixForGrad) ;
	// printf("Time taken for grad effic: %.2fs\n", (double)(clock() - tStartGradEffic)/CLOCKS_PER_SEC);
	
	// cv::Mat resultGrad;cv::add(delta,-delta1,resultGrad);
	// std::cout<<cv::sum(delta)<<" and "<<cv::sum(delta1)<<std::endl;
	// std::cout<<cv::sum(resultGrad)<<std::endl;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// fin test de div_eff et grad_eff




	// int sizeCh[3]= {10,10,10};
	// // int sizekh[4]= {10,10,10,10};
	// // cv::Mat aaa(4,sizekh,CV_64FC1,1.0);
	// cv::Mat a(3,sizeCh,CV_64FC1,1.0);
	// cv::Mat b=2*a;
	// std::cout<<cv::sum(a)<<std::endl;
	// std::cout<<cv::sum(b)<<std::endl;
	// cv::Mat aaa(3,sizeCh1,CV_64FC1,0.0);
	
	// cv::accumulateProduct(MatchingAlgorithm::getRow(aaa,0),MatchingAlgorithm::getRow(aaa,0),gap_per_pixel);cv::accumulateProduct(-MatchingAlgorithm::getRow(aaa,0),MatchingAlgorithm::getRow(aaa,0),gap_per_pixel);
	// std::cout<<cv::sum(gap_per_pixel)<<std::endl;

	// cv::Mat m = cv::Mat::ones(image1GrayDouble, 2,image1Gray.type());
	// int size[4] = {10,10,10,10};
	// cv::Mat M(4, size, CV_64FC1, 1.0);
	// cv::Mat h=MatchingAlgorithm::getRow(M,0);
	// cv::Mat h1=MatchingAlgorithm::getRow(h,0);
	// cv::Mat h2=MatchingAlgorithm::getRow(h1,0);
	// std::cout<<cv::sum(M)<<std::endl;
	// std::cout<<cv::sum(h)<<std::endl;
	// std::cout<<cv::sum(h1)<<std::endl;
	// std::cout<<cv::sum(h2)<<std::endl;

	// cv::randu(M,1.0,3.0);
	// cv::Mat MatToCompare(2, size, CV_64FC1, 2.0);
	// // cv::Mat result;//(2, size, CV_64FC1, 0.0);
	// // cv::compare(M,2.0,result,3);
	// cv::Mat result = -M < -2.0;
	// result.convertTo(result, CV_64FC1);
	// for (int i=0;i<M.size[0];i++)
	// 			{
	// 				std::cout<<"\n"<<std::endl;
	// 				for (int j=0;j<M.size[1];j++)
	// 					{
	// 							// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
	// 						// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
	// 						std::cout<<"value of pixel"<<i <<" and "<<j<<"for result : "<< result.at<double>(i,j)<<" and for original : "<< M.at<double>(i,j)<<"  "<<std::endl;
	// 					}
	// 	}
	// cv::Size a=M.size();
	// std::cout<<M.size[0]<<M.size[1]<<M.size[2]<<M.size[3]<<std::endl;
	// // cv::multiply(M.row(0),cv::eye(3,size,CV_64FC1,3.0),M.row(0));
	// cv::Mat A=M.t();
	// cv::Mat M0=(M.t()).row(0);
	// cv::Mat M1=(M.t()).row(1);
	// cv::Mat M2=(M.t()).row(2);
	// // cv::Mat M3=M.row(3);
	// double result=cv::norm(M);
	// std::cout<<result<<std::endl;
	// result=cv::norm(M0);std::cout<<result<<std::endl;
	// result=cv::norm(M1);std::cout<<result<<std::endl;
	// result=cv::norm(M2);std::cout<<result<<std::endl;

	// for (int i = 0; i < 100; i++)
	//  	for (int j = 0; j < 100; j++)
 	//    		for (int k = 0; k < 3; k++) 
	// 				std::cout<<"i,j,k : "<<i<<j<<k<<" "<<M.at<double>(i,j,k)<<std::endl;

	// std::cout<<
	// theAlgorithm.helpDebug();

	// theAlgorithm.showImages();
	// theAlgorithm.printContentsImage1();
	// printf(5.0)
	// cv::imshow("Output Image", image1Gray);
	// cv::waitKey(0);
	return EXIT_SUCCESS;
}
					

