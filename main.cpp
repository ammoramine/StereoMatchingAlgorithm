#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include "MatchingAlgorithm.h"
#include <getopt.h>
#include <unistd.h>
static int verbose_flag;

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// extern char *optarg;
// extern int optind, opterr, optopt;
void read_option(int argc, char* argv[],cv::Mat &image1,cv::Mat &image2)
{
	int c;
	while (1)
    {
      static struct option long_options[] =
        {
          /* These options set a flag. */
          {"verbose", no_argument,       &verbose_flag, 1},
          {"brief",   no_argument,       &verbose_flag, 0},
          /* These options don’t set a flag.
             We distinguish them by their indices. */
         
          {"im1",  required_argument, 0, 'a'},
          {"im2",  required_argument, 0, 'b'},
          {"Niter",  optional_argument, 0, 'c'},
          {0, 0, 0, 0}
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;

      c = getopt_long (argc, argv, "a:b:",
                       long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
        break;

      switch (c)
        {
        // case 1:
        //   printf("this algorithm compute the disparity map of two rectified image according to the article ''GLOBAL SOLUTIONS OF VARIATIONAL MODELS WITH CONVEX REGULARIZATION'', it needs the following syntax : \n ./exec -im1 path_to_image1 -im2 path_to_image2 " );
        case 0:
          /* If this option set a flag, do nothing else now. */
          if (long_options[option_index].flag != 0)
            break;
          printf ("option %s", long_options[option_index].name);
          if (optarg)
            printf (" with arg %s", optarg);
          printf ("\n");
          break;
        case 'a':
          printf ("path to image 1 `%s'\n", optarg);
          image1=cv::imread(optarg);
          break;

        case 'b':
          printf ("path to image 2 `%s'\n", optarg);
          image2=cv::imread(optarg);
          break;
        // case 'c':
        //   printf ("number of maximal iteration `%s'\n", optarg);
        //   Niter=atoi(optarg);
        //   break;
        case '?':
          /* getopt_long already printed an error message. */
          break;

        default:
          abort ();
        }
    }
    /* Instead of reporting ‘--verbose’
     and ‘--brief’ as they are encountered,
     we report the final status resulting from them. */
  if (verbose_flag)
  	printf("This algorithm compute the disparity map of two rectified image according to the article ''GLOBAL SOLUTIONS OF VARIATIONAL MODELS WITH CONVEX REGULARIZATION'', it needs the following syntax : \n ./exec -im1 path_to_image1 -im2 path_to_image2 \n" );
    // puts ("verbose flag is set");

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
    {
      printf ("non-option ARGV-elements: ");
      while (optind < argc)
        printf ("%s ", argv[optind++]);
      putchar ('\n');
    }

}

int main(int argc, char* argv[])
{
	

		// Hello();
	// char *cvalue = NULL;
	// getopt(argc,argv,"im2:");
	// // std::cout<<argv[optind]<<std::endl;
	// cvalue=optarg;
	// // std::cout<<cvalue<<std::endl;
	// printf ("cvalue = %s\n",cvalue);
	// optind=1;
	// std::cout<<optstring<<std::endl;
	// getopt(argc,argv,"-im1:");
	// std::cout<<argv[optind]<<std::endl;
	

	cv::Mat image1;//=cv::imread("input_pair/rectified_ref.tif");//, cv::IMREAD_LOAD_GDAL);
	cv::Mat image2;//=cv::imread("input_pair/rectified_sec.tif");//, cv::IMREAD_LOAD_GDAL);
	read_option(argc,argv,image1,image2);
	cv::Mat image1Gray;
	cv::Mat image2Gray;
	cv::Mat image1GrayDouble;
	cv::Mat image2GrayDouble;
	cv::cvtColor(image1, image1Gray, CV_RGB2GRAY);
	cv::cvtColor(image2, image2Gray, CV_RGB2GRAY);
	image1Gray.convertTo(image1GrayDouble, CV_64FC1);
	image2Gray.convertTo(image2GrayDouble, CV_64FC1);
	// cv::Mat m = cv::Mat::ones(2, 2,image1Gray.type());
	// std::cout<<m<<std::endl;
	// std::cout<< image1Gray.type()<<std::endl;
	// cv::namedWindow("Output Image");
	// cv::Size a=image1.size();
	// std::cout<<"width :"<<a.width<<"height : "<<a.height<<std::endl;
	MatchingAlgorithm theAlgorithm = MatchingAlgorithm(image1GrayDouble,image2GrayDouble);


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
					

