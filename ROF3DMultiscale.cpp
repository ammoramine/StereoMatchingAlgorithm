#include "ROF3DMultiscale.h"

// ROF3DMultiscale::ROF3DMultiscale(const std::vector<DataTerm>& data_terms,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double offset,double ratioGap,double precision) 
// // : m_nbMaxThreads(nbMaxThreads),m_offset(offset),m_precision(precision)
// // inputs: there should be at least two terms on the vector data_terms

// //this function resolve argmin_{v}( Sigma g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+Sigma |v(i,j+1,k)-v(i,j,k)|+Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma |v(i,j,k)-m_f(i,j,k)|^2) with v(i,j,k) but with a multiscale approach
// // we modify only the scale of the disparity, we multiply each time the data term by the value on the vector of scales, computed from the ratio of the different data_terms


// {
// 	// first initializaiton
// 	ROF3DMultiscale(data_terms[0],Niter,path_to_disparity,path_to_initial_disparity,nbMaxThreads,offset,ratioGap);
// 	for (int i=0;i<m_scales.size;i++)
// 	{
// 		ROF3DMultiscale(data_terms[i+1],m_Niter,m_path_to_disparity,m_nbMaxThreads,offset,ratioGap,m_x1CurrentCurrent,m_x2Current,m_x3Current);
// 	}
// 	// here we should get the x_i terms update them, and use the version of ROF3D with the xi

	
// }

ROF3DMultiscale::ROF3DMultiscale(const std::vector<DataTerm>& data_terms,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double ratioGap) : ROF3D(data_terms[0],10,addSuffixFloatBeforeExtension(path_to_disparity,0),path_to_initial_disparity,nbMaxThreads,ratioGap*0.001)
{
			// ROF3D(const DataTerm & dataTerm,int Niter,const std::string &path_to_disparity,const std::string &path_to_initial_disparity,size_t nbMaxThreads,double ratioGap,double precision=0.0000001);

	// computeScales();
	// printContentsOf3DCVMat(data_terms[0],true,"data_terms0");
	// printContentsOf3DCVMat(data_terms[1],true,"data_terms1");
	cv::Mat x1Current_doubleTemp,x2Current_doubleTemp,x3Current_doubleTemp;
	for (int i=1;i<data_terms.size();i++)
	{

		// changeScale(m_x1Current,x1Current_doubleTemp,m_scales[1]+1);
		// changeScale(m_x2Current,x2Current_doubleTemp,m_scales[1]+1);
		// changeScale(m_x3Current,x3Current_doubleTemp,m_scales[1]+1);

		changeScale(m_x1Current,x1Current_doubleTemp,data_terms[i].matrix);
		// printContentsOf3DCVMat(m_x1Current,true,"m_x1Current");
		// printContentsOf3DCVMat(x1Current_doubleTemp,true,"x1Current_doubleTemp");
		changeScale(m_x2Current,x2Current_doubleTemp,data_terms[i].matrix);
		changeScale(m_x3Current,x3Current_doubleTemp,data_terms[i].matrix);

		// ROF3D(data_term,int Niter,const std::string &path_to_disparity,size_t nbMaxThreads,double ratioGap,const cv::Mat &x1Current,const cv::Mat &x2Current,const cv::Mat &x3Current,double precision=0.0000001);

		// int found=path_to_disparity.find_first_of(".");
		// std::string m_path_to_disparityRefined=path_to_disparity;
		// m_path_to_disparityRefined.insert(found,std::to_string(i));
	// std::string path_to_disparity_reverse=path_to_disparity;
	// path_to_disparity_reverse.insert(found,"_reverse");
		DataTerm refinedDataTerm=data_terms[i];
		if (i!=data_terms.size()-1)
		{
			ROF3D rof=ROF3D(refinedDataTerm,30,addSuffixFloatBeforeExtension(path_to_disparity,i),m_nbMaxThreads,ratioGap,x1Current_doubleTemp,x2Current_doubleTemp,x3Current_doubleTemp);
			rof.getPrimal(m_x1Current,m_x2Current,m_x3Current);
		}
		else
		{
			ROF3D rof=ROF3D(refinedDataTerm,Niter,path_to_disparity,m_nbMaxThreads,ratioGap,x1Current_doubleTemp,x2Current_doubleTemp,x3Current_doubleTemp);

		}
		// m_x1Current=rof.m_x1Current;m_x2Current=rof.m_x2Current;m_x3Current=rof.m_x3Current;

		// ROF3D rof=ROF3D(refinedDataTerm,Niter,m_path_to_disparityRefined,path_to_initial_disparity,nbMaxThreads,ratioGap);
		// rof.computeDisparityFromPrimal(m_path_to_disparityRefined);
		// ROF3D::ROF3D(const DataTerm & dataTerm,int Niter,const std::string &path_to_disparity,size_t nbMaxThreads,double ratioGap,const cv::Mat &x1Current,const cv::Mat &x2Current,const cv::Mat &x3Current,double precision) : m_nbMaxThreads(nbMaxThreads),m_precision(precision)

		// ROF3D(const cv::Mat & data_term,int Niter,const std::string &path_to_disparity,size_t nbMaxThreads,double offset,double ratioGap,const cv::Mat &x1Current,const cv::Mat &x2Current,const cv::Mat &x3Current,double precision=0.0000001);
	}
	// int size[3]={m_g.size[0],m_g.size[1],2*m_g.size[2]};
	
	// cv::Mat m_x1Current_double;//=cv::Mat(3,size,CV_64FC1,0.0);

	
	// printContentsOf3DCVMat(m_x1Current,true,"m_x1Current");
	// printContentsOf3DCVMat(m_x1Current_double,true,"m_x1Current_double");
	// int i=10;
	// cv::Mat imageToShow;cv::Mat imageToShowBigger;getRow3D(m_x1Current,i,imageToShow);getRow3D(m_x1Current_double,i,imageToShowBigger);
	// imageToShow.convertTo(imageToShow,CV_32FC1);imageToShowBigger.convertTo(imageToShowBigger,CV_32FC1);
 //    iio_write_image_float("image.tif",(float *)imageToShow.data,imageToShow.size[1],imageToShow.size[0]);
 //    iio_write_image_float("imageToShow.tif",(float *)imageToShowBigger.data,imageToShowBigger.size[1],imageToShowBigger.size[0]);


}
// void ROF3DMultiscale::computeScales()
// {
// 	for (int i=0;i<m_data_terms.size();i++)
// 	{
// 		m_scales.push_back(m_data_terms[i].size[2]);
// 	}
// }
// void ROF3DMultiscale::changeScale(const cv::Mat &matrix3DOld,cv::Mat &matrix3DNew,int scale)
// {
// 	//change the scale of the last term of matrix3D, we multiply by 2
// 	int size[3]={m_g.size[0],m_g.size[1],scale};
// 	matrix3DNew=cv::Mat(3,size,CV_64FC1,0.0);
// 	cv::Mat matrix3DOldi;cv::Mat matrix3DNewi;
// 	// printContentsOf3DCVMat(matrix3DOld,true,"matrix3DOld");
// 	// printContentsOf3DCVMat(matrix3DNew,true	,"matrix3DNew");
// 	for (int i=0;i<matrix3DOld.size[0];i++)
// 	{	
// 		getRow3D(matrix3DOld,i,matrix3DOldi);getRow3D(matrix3DNew,i,matrix3DNewi);
// 		// cv::Size newSize=cv::Size(matrix3DOldi.size[0],2*matrix3DOldi.size[1]);
// 		// cv::Size newSize=matrix3DOldi.size();newSize.width=4*newSize.width;
// 		cv::resize(matrix3DOldi,matrix3DNewi,matrix3DNewi.size());
// 		// printContentsOf3DCVMat(matrix3DOldi,true,"matrix3DOldi");
// 		// printContentsOf3DCVMat(matrix3DNewi,true,"matrix3DNewi");
// 		setRow3D(matrix3DNewi,i,matrix3DNew);
// 		cv::Mat matrix3DNewiCopy;getRow3D(matrix3DNew,i,matrix3DNewiCopy);
// 		// printContentsOf3DCVMat(matrix3DNewiCopy,true,"matrix3DNewiCopy");

// 	}
// 	// printContentsOf3DCVMat("")
// }

void ROF3DMultiscale::changeScale(const cv::Mat &matrix3DOld,cv::Mat &matrix3DNew,const cv::Mat &newDataTerm)
{
	//change the evolving arguments of Xi using the values of the dataTerm, remember that ithe size in the last dimension is above the one of the dataTerm by one. 
	int size[3]={newDataTerm.size[0],newDataTerm.size[1],newDataTerm.size[2]+1};// there is a +1 in the last dimension on the primal variables
	int sizeTemp[3]={matrix3DOld.size[0],newDataTerm.size[1],newDataTerm.size[2]+1};// there is a +1 in the last dimension on the primal variables
	matrix3DNew=cv::Mat(3,size,CV_64FC1,0.0);

	cv::Mat matrix3DNewTemp=cv::Mat(3,sizeTemp,CV_64FC1,0.0);
	cv::Mat matrix3DOldi;cv::Mat matrix3DNewTempi;
	// printContentsOf3DCVMat(matrix3DOld,true,"matrix3DOld");
	// printContentsOf3DCVMat(matrix3DNew,true	,"matrix3DNew");
	for (int i=0;i<matrix3DNewTemp.size[0];i++)
	{	
		getRow3D(matrix3DOld,i,matrix3DOldi);
		getRow3D(matrix3DNewTemp,i,matrix3DNewTempi);
		// cv::Size newSize=cv::Size(matrix3DOldi.size[0],2*matrix3DOldi.size[1]);
		// cv::Size newSize=matrix3DOldi.size();newSize.width=4*newSize.width;
		cv::resize(matrix3DOldi,matrix3DNewTempi,matrix3DNewTempi.size());
		// printContentsOf3DCVMat(matrix3DOldi,true,"matrix3DOldi");
		// printContentsOf3DCVMat(matrix3DNewTempi,true,"matrix3DNewTempi");
		setRow3D(matrix3DNewTempi,i,matrix3DNewTemp);
		cv::Mat matrix3DNewTempiCopy;getRow3D(matrix3DNewTemp,i,matrix3DNewTempiCopy);
		// printContentsOf3DCVMat(matrix3DNewTempiCopy,true,"matrix3DNewTempiCopy");

	}
	cv::Mat matrix3DNewTempk,matrix3DNewk;
	for (int k=0;k<matrix3DNew.size[2];k++)
	{	
		getLayer3DBeta(matrix3DNewTemp,k,matrix3DNewTempk);getLayer3DBeta(matrix3DNew,k,matrix3DNewk);// this last instruction is necessary just to get the size
		// printContentsOf3DCVMat(matrix3DNewTempk,true,"matrix3DNewTempk");
		// printContentsOf3DCVMat(matrix3DNewk,true,"matrix3DNewk");
		cv::resize(matrix3DNewTempk,matrix3DNewk,matrix3DNewk.size());
		// printContentsOf3DCVMat(matrix3DOldi,true,"matrix3DOldi");
		// printContentsOf3DCVMat(matrix3DNewk,true,"matrix3DNewk");
		setLayer3D(matrix3DNewk,k,matrix3DNew);
		cv::Mat matrix3DNewkCopy;getLayer3DBeta(matrix3DNew,k,matrix3DNewkCopy);
		// printContentsOf3DCVMat(matrix3DNewkCopy,true,"matrix3DNewkCopy");

	}
	// printContentsOf3DCVMat("")
}