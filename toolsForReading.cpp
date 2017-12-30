#include "toolsForReading.h"

//// global table of the data_term

const char *global_table_of_data_term[] = {"absdiff","census"};
const char *method[]={"direct","accelerated"};


std::string compare_on_list(char * option,const char* listOfElements[],int sizeOflistOfElements,char * messageToPrint)
// this function search on the a list of 
{
  // int sizeOfElements=sizeof(listOfElements)/sizeof(char*);
  // int r=sizeOfElements;
   for(int i=0; i<sizeOflistOfElements; i++)
      if (strcmp(option,listOfElements[i])==0)
      { 
        return std::string(listOfElements[i]);
      }
  // if(r==sizeOfElements)
  //   { 
      throw std::invalid_argument( messageToPrint );
    // }
}


void read_option(int argc, char* argv[],cv::Mat &image1,cv::Mat &image2,std::string  &data_term_option,int &tsize,double &offset,double &ratioGap,int &Niter,double &zoom,std::string &path_to_disparity,std::string &path_to_initial_disparity,int &nbmaxThreadPoolThreading,std::string &method,bool &multiScale)
{
	int c;
  //
  path_to_initial_disparity="";
  ratioGap=0.0;
  zoom=1;
  multiScale=false;
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
          {"dataterm",  required_argument, 0, 'c'},
          {"Niter",  required_argument, 0, 'd'},
          {"tsize",  required_argument, 0, 'e'},
          {"offset",  required_argument, 0, 'f'},
          {"path_to_disparity",  required_argument, 0, 'g'},
          {"threadsMax",required_argument,0,'h'},
          {"method",required_argument,0,'i'},
          {"path_to_initial_disparity",required_argument,0,'j'},
          {"ratioGap",required_argument,0,'k'},
          {"zoom",required_argument,0,'l'},
          {"multiscale",no_argument,0,'m'},
          {0, 0, 0, 0}
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;

      c = getopt_long (argc, argv, "a:b:c:d:e:f:g:h:i:j::",
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
          printf ("path to image 1 :image to which the computed disparity is applied ,(disparities computed from image 1 to image 2) `%s'\n", optarg);
          readAndConvertImageToGray(optarg,image1);
          // image1=cv::imread(optarg,cv::IMREAD_LOAD_GDAL);
          // image1.convertTo(image1,CV_32FC1);
          // iio_write_image_float("image1_gray.tif",(float*)image1.clone().data,image1.clone().size[1],image1.clone().size[0]);

          break;

        case 'b':
          printf ("path to image 2 :target image with whom the first image should be matched,(disparities computed from image 1 to image 2) `%s'\n", optarg);
          readAndConvertImageToGray(optarg,image2);
          // image2=cv::imread(optarg,cv::IMREAD_LOAD_GDAL);
          // image2.convertTo(image2,CV_32FC1);
          // iio_write_image_float("image2_gray.tif",(float*)image2.data,image2.size[1],image2.size[0]);
          
          break;
        case 'c':
        {
          // int i=
          // assert (get_dataterm_index(std::string(optarg))!=NULL && "couldn't recognize the data term, choose among the following ");
          data_term_option=compare_on_list(optarg,global_table_of_data_term,sizeof(global_table_of_data_term)/sizeof(char*),"incorrect data term");
          std::cout<<"data_term used : "<<data_term_option<<std::endl;

          }break;
        case 'd':
          printf ("number of maximal iterations `%s'\n", optarg);
          Niter=atoi(optarg);
          break;
        case 'e':
          printf ("length of the interval of disparity, `%s'\n", optarg);
          tsize=atoi(optarg);
          break;
        case 'f':
          printf ("offset is the smallest algebrical value of the disparity, disparities computed from the left to the right`%s'\n", optarg);
          offset=atof(optarg);
          break;
        case 'g':
          printf ("prefix of the name of the disparity image`%s'\n", optarg);
          path_to_disparity=std::string(optarg);
          break;
        case 'h':
          printf ("maximal number of threads used in pool threading used`%s'\n", optarg);
          nbmaxThreadPoolThreading=atoi(optarg);
          break;
        case 'i':
          printf ("method used`%s'\n", optarg);
          method=std::string(optarg);
          break;
        case 'j':
          // if (optarg!=NULL)
          // {
            printf ("initial disparity used and path to the image is  `%s'\n", optarg);
            path_to_initial_disparity=std::string(optarg);
        case 'k':
            printf("ratio between the intial primal dual gap and the current primal dual gap before stopping is '%s'\n",optarg);
            ratioGap=atof(optarg);
            if (ratioGap>1.0 or ratioGap<0.0)
            {
                  throw std::invalid_argument( "the ratio gap should be between 0 and 1" );

            }
        case 'l':
          // if (optarg!=NULL)
          // {
            printf ("step of the disparity is the inverse of the value of the zoom,which is equal to : `%s'\n", optarg);
            printf("it should be of the form 2^{n} with n in Z");
            zoom=atof(optarg);
          // }
          // else
          // {
            // path_to_initial_disparity="";
          // }
          break;
        case 'm':
          // if (optarg!=NULL)
          // {
            printf ("multiscale strategy used for the computation of the disparity : `%s'\n", optarg);
            multiScale=true;
          // }
          // else
          // {
            // path_to_initial_disparity="";
          // }
          break;
                  // {"initialDisparity",optional_argument,0,'j'},

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
void readAndConvertImageToGray(const std::string &pathToImage,cv::Mat &output)
{

  cv::Mat outputTemp=cv::imread(pathToImage,cv::IMREAD_LOAD_GDAL);
  int nbChannels=outputTemp.channels();
  // printContentsOf3DCVMat(output,true,"output");
  // cv::Mat outputGrayDouble;
  if (nbChannels==3)
  {
    cv::Mat outputGray;
    cv::cvtColor(outputTemp, outputGray, CV_RGB2GRAY);
    outputGray.convertTo(output, CV_64FC1);
  }
  else if (nbChannels==1)
  {
    outputTemp.convertTo(output, CV_64FC1);
  }
  else
  {
    throw std::invalid_argument( "this kind of image is not managed in the code ..." );

  }
  // printContentsOf3DCVMat(output,true,"output32");
  // cv::imread(optarg,cv::IMREAD_LOAD_GDAL);

}

