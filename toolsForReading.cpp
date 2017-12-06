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


void read_option(int argc, char* argv[],cv::Mat &image1,cv::Mat &image2,std::string  &data_term_option,int &tsize,double &offset,int &Niter,std::string &path_to_disparity,int &nbmaxThreadPoolThreading,std::string &method)
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
          {"dataterm",  required_argument, 0, 'c'},
          {"Niter",  required_argument, 0, 'd'},
          {"tsize",  required_argument, 0, 'e'},
          {"offset",  required_argument, 0, 'f'},
          {"path_to_disparity",  required_argument, 0, 'g'},
          {"threadsMax",required_argument,0,'h'},
          {"method",required_argument,0,'i'},
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
          printf ("path to image 1 (image on the right) `%s'\n", optarg);
          readAndConvertImageToGray(optarg,image1);
          // image1=cv::imread(optarg,cv::IMREAD_LOAD_GDAL);
          break;

        case 'b':
          printf ("path to image 2 (image on the left )`%s'\n", optarg);
          readAndConvertImageToGray(optarg,image2);
          // image2=cv::imread(optarg,cv::IMREAD_LOAD_GDAL);
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
          printf ("length of the interval of disparity `%s'\n", optarg);
          tsize=atoi(optarg);
          break;
        case 'f':
          printf ("offset is the smallest algebrical value of the disparity`%s'\n", optarg);
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
  // cv::imread(optarg,cv::IMREAD_LOAD_GDAL);

}

