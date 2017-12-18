#include "census.h"

Census::Census(const cv::Mat &image1,const cv::Mat &image2,cv::Mat &data_term,double offset) : m_image1(image1),m_image2(image2),m_offset(offset) //normally g is preallocated
{
	// printContentsOf3DCVMat(image1,true,"image1");
	// printContentsOf3DCVMat(image2,true,"image2");
	m_dataterm=data_term;
	computeDataTerm();
	data_term=m_dataterm;
	// m_disparityExtent=g.size[2];
}

double Census::hammingDistance(const cv::Mat &ternarySignature1,const cv::Mat &ternarySignature2)
{
	double result=0.0;
	int lengthArray=ternarySignature1.size[0];
		// printContentsOf3DCVMat(ternarySignature1,true,"ternarySignature1");
		// printContentsOf3DCVMat(ternarySignature2,true,"ternarySignature2");

	for (int i=0;i<lengthArray;i++)
	{
		if(ternarySignature1.at<double>(i)!=ternarySignature2.at<double>(i))
		{
			result+=1;
		}
	}
	return result;
}
void Census::computeCensusSignature(const cv::Mat &image, cv::Mat &ternaryCensusSignature)
{
	int size[3]={image.size[0],image.size[1],25}; //8 corresponds to the dmimension of the beighborhood-1, here it's the square around the pixel
	ternaryCensusSignature=cv::Mat(3,size,CV_64FC1,0.0);
	cv::Mat ternaryCensusSignaturei;
	cv::Mat ternaryCensusSignatureij;
	cv::Mat intensityNeighbors;
	cv::Mat imagei;

	// int sizePaddedImage={size[0]+2,size[1]+2,size[3]};
	cv::Mat paddedImage;//=cv::Mat(3,sizePaddedImage,CV_64FC1,0.0);
	int radiusWindow=2;
	cv::copyMakeBorder(image,paddedImage,radiusWindow,radiusWindow,radiusWindow,radiusWindow,cv::BORDER_CONSTANT,cv::Scalar(0));
	// cv::Mat paddedImagei;
	// printContentsOf3DCVMat(image,true,"image");
	// printContentsOf3DCVMat(paddedImage,true,"paddedImage");
	// double intensityNeighbors[8];
	for (int i=0;i<size[0];i++)
	{
		getRow3D(ternaryCensusSignature,i,ternaryCensusSignaturei);
		getRow3D(paddedImage,i+radiusWindow,imagei); //row  i of image is row i+radiusWindow of paddedImage

		for (int j=0;j<size[1];j++)
		{
			double * ternaryCensusSignatureij =ternaryCensusSignaturei.ptr<double>(j);
			intensityNeighbors=paddedImage(cv::Range(i-2+radiusWindow,i+3+radiusWindow),cv::Range(j-2+radiusWindow,j+3+radiusWindow));
			//pixel (i,j) from image is pixel (i+radiusWindow,j+radiusWindow) of the paddedImage
			double Icur=imagei.at<double>(j+radiusWindow);
			// same justification
			intensityNeighbors=intensityNeighbors.clone().reshape(0,1);
			// printContentsOf3DCVMat(intensityNeighbors,true,"intensityNeighbors");
			for (int k=0;k<size[2];k++)
			{
				double a=intensityNeighbors.at<double>(k)-Icur;
				if (a < 0)
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

void Census::computeDataTerm()
{
	// we suppose that the cv::Mat object have been initialized before to the value  g=cv::Mat(3, size, CV_64FC1, 500.0); with size linked to the size of images: image1 and image2.

	 //the convention y,x,t
	// the element of index i,j,k is associated to the pixels (i,j) of the first image, and (i,j+(k+offset)) on the second image ( target image), if the index can't be reached, we associate to the dataTerm an infinite value
	//where the offset is the minimal value of the disparity and k goes throught the interval of the disparity.

	//So that to recover the disparity, from an index (i,j,k), we should add the offset to the value of k 

	// Convention: the disparity is positive when we move the image to the right, and  negative in the other direction, algebrically the disparity is the algebrical distance from the first image to the second one

	// g=cv::Mat(3, size, CV_64FC1, 0.0);
	// cv::Mat ternaryCensusSignature1;
	// cv::Mat ternaryCensusSignature2;
	int intOffset=int(floor(m_offset));
	cv::Mat m_CensusSignature_1i;
	cv::Mat m_CensusSignature_2i;
	cv::Mat m_datatermi;
	
	cv::Mat m_CensusSignature_1ij;
	cv::Mat m_CensusSignature_2ij;
	cv::Mat m_datatermij;

	cv::Mat m_CensusSignature_2ijpk;
	// m_image1=image1;
	// m_image2=image2;
	computeCensusSignature(m_image1,m_CensusSignature_1);
	computeCensusSignature(m_image2,m_CensusSignature_2);
	// cv::Mat censusSignature_1=m_CensusSignature_1.clone();
	// cv::Mat censusSignature_2=m_CensusSignature_2.clone();

	// printContentsOf3DCVMat(m_CensusSignature_1,true,"m_CensusSignature_1");
	
	// printContentsOf3DCVMat(m_CensusSignature_1,true,"m_CensusSignature_1");
	// printContentsOf3DCVMat(m_CensusSignature_2,true,"m_CensusSignature_2");


		for (int i=0;i<m_dataterm.size[0];i++)
		{
			// double * deltaxPtr=deltax.ptr<double>(0);
			// const double * image1iPtr= image1.ptr<double>(i);
			// const double * image2iPtr= image2.ptr<double>(i);
			getRow3D(m_CensusSignature_1,i,m_CensusSignature_1i);
			getRow3D(m_CensusSignature_2,i,m_CensusSignature_2i);
			
			// printContentsOf3DCVMat(m_CensusSignature_1i,true,"m_CensusSignature_1i");
			// printContentsOf3DCVMat(m_CensusSignature_2i,true,"m_CensusSignature_2i");
			
			getRow3D(m_dataterm,i,m_datatermi);
			for (int j=0;j<m_dataterm.size[1];j++)
			{
				getRow2D(m_CensusSignature_1i,j,m_CensusSignature_1ij);
				getRow2D(m_datatermi,j,m_datatermij);
				
				//mink and maxk are the extremal values of the disparity
				int maxk=std::min(intOffset+m_dataterm.size[2]-1,(m_CensusSignature_2i.size[0]-1)-j);
				int mink=std::max(intOffset,-j);
				for(int k=mink;k<=maxk;k++)
				{
					// double * m_CensusSignature_2ijmk =m_CensusSignature_2i.ptr<double>(j-k);
					getRow2D(m_CensusSignature_2i,j+k,m_CensusSignature_2ijpk);
			// for (int km
					// printContentsOf3DCVMat(m_CensusSignature_1ij,true,"m_CensusSignature_1ij");
					// printContentsOf3DCVMat(m_CensusSignature_2ijmk,true,"m_CensusSignature_2ijmk");

					m_datatermij.at<double>(k-intOffset)=hammingDistance(m_CensusSignature_1ij,m_CensusSignature_2ijpk);// we should remove the offset, because there is no negative index on the opencv Objects

				}
				// printContentsOf3DCVMat(m_datatermij,true,"m_datatermij");
				// printContentsOf3DCVMat(m_datatermij,true,"m_datatermij");
				// delete m_datatermij;
			}
			// printContentsOf3DCVMat(m_datatermi,true,"m_datatermi");
		}
		// printContentsOf3DCVMat(m_dataterm,true,"m_dataterm");
			// delete m_image1iPtr;
			// delete m_image2iPtr;
}