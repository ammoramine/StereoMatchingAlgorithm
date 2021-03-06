#include "someTools.h"

cv::Mat getLayer(cv::Mat Matrix3D,int layer_number)
// get a copy of the external layer of a 3D Matrix
{
	int size[2] = { (Matrix3D.size()).height, (Matrix3D.size()).width};
	cv::Mat layer(2, size, CV_64FC1, 0.0);
	cv::Mat matrix3Di;cv::Mat layeri;
	for (int i=0;i<size[0];i++)
				{
					getRow3D(Matrix3D,i,matrix3Di);
					getRow2D(layer,i,layeri);
				for (int j=0;j<size[1];j++)
					{
					// for (int k=0;k<m_t_size,k++)
					// 	{
							layeri.at<double>(j)=matrix3Di.at<double>(j,layer_number);
						// }
					}
				}
	return layer;
}



cv::Mat getLayer2DOld(const cv::Mat &matrix2D,int layer_number)
// layer_number should be below Matrix2D.size[1]
{
	// int sizeMatrix2D[2] = { matrix2D.size[0], matrix2D.size[1]};
	// int stepMatrix2D[2] = { matrix2D.step[0], matrix2D.step[1]};

	int sizeLayer[1] = { matrix2D.size[0]};
	cv::Mat layer=cv::Mat(1, sizeLayer, CV_64FC1, 0.0);

	uchar * matrix2Ddata=matrix2D.data;
	// // // uchar layerdata[sizeof(matrix2Ddata)];
	// // // layer.data;

	int step= matrix2D.step[0];
	int limit=matrix2D.step[0]*matrix2D.size[0]+ matrix2D.step[1]*layer_number;
	
	int stepLayer=0;
	// // memcpy(layer.data+0,matrix2Ddata,sizeof(double));

	for(int k=matrix2D.step[1]*layer_number;k < limit;k+=step)
	{
		// layerdata
		// layer.data=matrix2Ddata+k;
		// stepLayer++;
		// memcpy(layer.data+stepLayer,matrix2Ddata+k,sizeof(double));
		memcpy(layer.data+stepLayer*matrix2D.step[1],matrix2Ddata+k,sizeof(double));

		stepLayer++;
	}
 	// layer.data=matrix2D.col(layer_number).data;
 	return cv::Mat(1,sizeLayer,CV_64FC1,layer.data);
 	// return layer;
// addr(Mi0,...,iM.dims−1)=M.data+M.step[0]∗i0+M.step[1]∗i1+...+M.step[M.dims−1]∗iM.dims−1
}

void getLayer2D(const cv::Mat &matrix2D,int layer_number,cv::Mat &layer1D)
// layer_number should be below Matrix2D.size[1]
{
	// int sizeLayer[1] = { matrix2D.size[0]};
	// layer1D=cv::Mat(1, sizeLayer, CV_64FC1, 0.0);
	layer1D=matrix2D.col(layer_number);
	// matrix2D.col(layer_number).copyTo(layer1D);
	// return layer1D;
}

void testLayer2D()
{
	double m1[3][3] = { {10,1,2}, {3,4,5}, {6,7,8} };
	int size1[]={3,3};
	// // int sizeCh[3]= {50,50,50};
	cv::Mat matrix2D(2,size1,CV_64FC1,m1);
	// cv::randu(matrix2D,0,1.0);
	cv::Mat layer1;
	getLayer2D(matrix2D,0,layer1);
	// layer1.at<double>(0)=3.4;
	printContentsOf3DCVMat(matrix2D,false,"matrix2D");
	printContentsOf3DCVMat(layer1,false,"layer1");
	// cv::randu(layer1,-1.0,0.0);
	// printContentsOf3DCVMat(M1,true,"M1After");
	// printContentsOf3DCVMat(layer1,true,"layer1After");
}

void testLayer2DBis()
{
	double m1[3][3] = { {10,1,2}, {3,4,5}, {6,7,8} };
	int size1[]={3,3};
	// // int sizeCh[3]= {50,50,50};
	cv::Mat matrix2D(2,size1,CV_64FC1,m1);
	cv::Range arrayOfRanges[]={cv::Range(1,3),cv::Range(0,2)};
	cv::Mat layer1=matrix2D(arrayOfRanges);//.copyTo(layer1);
	// cv::randu(matrix2D,0,1.0);
	// cv::Mat layer1;
	// getLayer2D(matrix2D,0,layer1);
	// layer1.at<double>(0)=3.4;
	printContentsOf3DCVMat(matrix2D,false,"matrix2D");
	printContentsOf3DCVMat(layer1,false,"layer1");
	// cv::randu(layer1,-1.0,0.0);
	// printContentsOf3DCVMat(M1,true,"M1After");
	// printContentsOf3DCVMat(layer1,true,"layer1After");
}

void getLayer3D(const cv::Mat &matrix3D,int layer_number,cv::Mat &layer)
// returns a copy and not a reference, would be better if it was possible to return a reference
{
	int sizeLayer[2] = { matrix3D.size[0],matrix3D.size[1]};
	layer=cv::Mat(2, sizeLayer, CV_64FC1, 0.0);

	cv::Mat matrix3Di;cv::Mat layeri;
	// MatConstIterator<double> it,it_end,// = M.end<double>();
	for (int i=0;i<matrix3D.size[0];i++)
	{
		getRow3D(matrix3D,i,matrix3Di);
		getRow2D(layer,i,layeri);
		// double* layeri=layer.ptr<double>(i);
		// uchar * 
		// uchar* positionMatrix3D=matrix3D.data+matrix3Di.step[1]*layer_number;
		// for (int j=0;j<matrix3D.size[1];j++)
		// {
		// 	// positionMatrix3D+=matrix3Di.step[0];
		// 	layeri[j]=matrix3Di.at<double>(j,k);//*(matrix3D.data+matrix3Di.step[1]*layer_number+j*matrix3Di.step[0]);
		// }
		// printContentsOf3DCVMat(layeri,false);
		// layeri.data=matrix3Di.col(layer_number).data;
		cv::Mat tempLayeri;
		// getLayer2D(matrix3Di,layer_number,layeri);
		getLayer2D(matrix3Di,layer_number,tempLayeri);
		tempLayeri.copyTo(layeri);
		// .copyTo(layeri);
		// printContentsOf3DCVMat(layeri,false);
		// printContentsOf3DCVMat(layer,false);
	}
	// return layer;

}

void getLayer3DBeta(const cv::Mat &matrix3D,int layer_number,cv::Mat &layer1)
// returns a reference on the 2D matrix matrix3D(.,.,layer_number)
{

	cv::Range arrayOfRanges[]={cv::Range(0,matrix3D.size[0]),cv::Range(0,matrix3D.size[1]),cv::Range(layer_number,layer_number+1)};
	// layer1=matrix3D(arrayOfRanges);// there is a problem when we take consider a header to the data, but when we consider the copy the problem is resolved ... (specially when we use the at method), why ? ... 
	// cv::Mat layer1Temp;matrix3D(arrayOfRanges).copyTo(layer1Temp);
	cv::Mat layer1Temp=cv::Mat(matrix3D,arrayOfRanges);//layer1Temp.copyTo(layer1);
	// int sizeLayer[2] = { layer1Temp.size[0],layer1Temp.size[1]};
	// size_t stepLayer[2] = { layer1Temp.step[0],layer1Temp.step[1]};
	// layer1=cv::Mat(2,sizeLayer,layer1Temp.type(),layer1Temp.data,stepLayer);
	cast3DMatrixTo2DMatrix(layer1Temp,layer1);
	// int sizeLayer[2] = { matrix3D.size[0],matrix3D.size[1]};
	// layer1=layer1Temp.reshape(0,2,sizeLayer);
	// layer1=layer1Temp.reshape(matrix3D.size[0],matrix3D.size[1]);

}

void getLayer3DReference(const cv::Mat &matrix3D,int layer_number,cv::Mat &layer1)
// returns a reference on the 3D matrix matrix3D(.,.,layer_number), with a size of 1 on the last dimension
{

	cv::Range arrayOfRanges[]={cv::Range(0,matrix3D.size[0]),cv::Range(0,matrix3D.size[1]),cv::Range(layer_number,layer_number+1)};
	// layer1=matrix3D(arrayOfRanges);// there is a problem when we take consider a header to the data, but when we consider the copy the problem is resolved ... (specially when we use the at method), why ? ... 
	// cv::Mat layer1Temp;matrix3D(arrayOfRanges).copyTo(layer1Temp);
	cv::Mat layer1Temp=cv::Mat(matrix3D,arrayOfRanges);//layer1Temp.copyTo(layer1);
	cast3DMatrixTo2DMatrixKeepingReference(layer1Temp,layer1);
	// getLayer3DReference(layer1Temp,layer);
	// int sizeLayer[2] = { layer1Temp.size[0],layer1Temp.size[1]};
	// size_t stepLayer[2] = { layer1Temp.step[0],layer1Temp.step[1]};
	// layer1=cv::Mat(2,sizeLayer,layer1Temp.type(),layer1Temp.data,stepLayer);
	// cast3DMatrixTo2DMatrix(layer1Temp,layer1);
	// int sizeLayer[2] = { matrix3D.size[0],matrix3D.size[1]};
	// layer1=layer1Temp.reshape(0,2,sizeLayer);
	// layer1=layer1Temp.reshape(matrix3D.size[0],matrix3D.size[1]);

}

void testLayer3D()
{
	double m[3][3][3] = { {{0,1,2}, {3,4,5}, {6,7,8}}, {{10,11,12}, {13,14,15}, {16,17,18}},{{20,21,22}, {23,24,25}, {26,27,28}} };
	// // double m[3][3][3] = { {{0,0.5,1.0}, {1.0,1.5,2.0}, {2.0,2.5,3.0}}, {{0,0.5,1.0}, {1.0,1.5,2.0}, {2.0,2.5,3.0}} , {{0,0.5,1.0}, {1.0,1.5,2.0}, {2.0,2.5,3.0}} };
	int size[]={3,3,3};
	// // int sizeCh[3]= {50,50,50};
	cv::Mat M(3,size,CV_64FC1,m);
	cv::Mat layer;
	getLayer3DBeta(M,0,layer);
	// cv::Mat layerCopy;layer.copyTo(layerCopy);
	printContentsOf3DCVMat(M,false,"M.txt");
	// std::cout<<layerCopy.at<double>(0,0,2)<<std::endl;
	printContentsOf3DCVMat(layer,false,"layer.txt");	
	// printContentsOf3DCVMat(layerCopy,false,"layerCopy.txt");
	// cv::randu(layer,-1.0,0.0);
	// printContentsOf3DCVMat(M,true,"MAfter.txt");
	// printContentsOf3DCVMat(layer,true,"layerAfter.txt");
}



// cv::Mat getRow4D(const cv::Mat &Matrix4D,int numberRow,bool newOne)
// // get the  row numer numberRow from a 4D matrix
// {

// 	int dims[] = { Matrix4D.size[1], Matrix4D.size[2],Matrix4D.size[3]};
// 	if (numberRow > Matrix4D.size[0] or numberRow < 0)
// 		{
// 			throw std::invalid_argument( "received false row" );
// 		}
// 	cv::Mat extractedMatrix(3,dims, CV_64FC1, Matrix4D.data + Matrix4D.step[0] * numberRow);
// 	return extractedMatrix;
// }

 void getRow4D(const cv::Mat &Matrix4D,int numberRow,cv::Mat &extractedMatrix)
// get the  row numer numberRow from a 4D matrix, not a copy but a reference to the real row
{

	int dims[] = { Matrix4D.size[1], Matrix4D.size[2], Matrix4D.size[3]};
	if (numberRow > Matrix4D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	// cv::Range ranges[]={cv::Range(numberRow,numberRow+1),cv::Range(0,Matrix3D.size[1]),cv::Range(0,Matrix3D.size[2])};
	// extractedMatrix=cv::Mat(2,dims, CV_64FC1, Matrix3D(ranges).data);
	extractedMatrix=cv::Mat(3,dims, CV_64FC1, Matrix4D.data + Matrix4D.step[0] * numberRow);
}

 void getRow3D(const cv::Mat &Matrix3D,int numberRow,cv::Mat &extractedMatrix)
// get the  row numer numberRow from a 4D matrix, not a copy but a reference to the real row
{

	int dims[] = { Matrix3D.size[1], Matrix3D.size[2]};
	if (numberRow > Matrix3D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	// cv::Range ranges[]={cv::Range(numberRow,numberRow+1),cv::Range(0,Matrix3D.size[1]),cv::Range(0,Matrix3D.size[2])};
	// extractedMatrix=cv::Mat(2,dims, CV_64FC1, Matrix3D(ranges).data);
	extractedMatrix=cv::Mat(2,dims, CV_64FC1, Matrix3D.data + Matrix3D.step[0] * numberRow);
}


void getRow2D(const cv::Mat &Matrix2D,int numberRow,cv::Mat &extractedMatrix)
// get the  row numer numberRow from a 4D matrix
{
	int dims[] = { Matrix2D.size[1]};
	if (numberRow > Matrix2D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	extractedMatrix=cv::Mat(1,dims, CV_64FC1, Matrix2D.data + Matrix2D.step[0] * numberRow);
	// return extractedMatrix;
}



void setRow3D(const cv::Mat &Matrix3Di,int i,cv::Mat &Matrix3D)
{
	cv::Mat Matrix3DiOld;getRow3D(Matrix3D,i,Matrix3DiOld);

	cv::MatConstIterator_<double> itMat3Di=Matrix3Di.begin<double>();
	cv::MatConstIterator_<double>  itMat3DiEnd=Matrix3Di.end<double>();

	cv::MatIterator_<double> itMat3DiOld=Matrix3DiOld.begin<double>();
	cv::MatIterator_<double> itMat3DiOldEnd=Matrix3DiOld.end<double>();

	

	while ( itMat3Di!= itMat3DiEnd and itMat3DiOld!= itMat3DiOldEnd) {
		(*itMat3DiOld)=(*itMat3Di);
		itMat3DiOld++;itMat3Di++;
	}

}

void setLayer3D(const cv::Mat &Matrix3Dk,int k,cv::Mat &Matrix3D)
{
	// cv::Mat Matrix3DkOld;getLayer3DReference(Matrix3D,k,Matrix3DkOld);
	cv::Range arrayOfRanges[]={cv::Range(0,Matrix3D.size[0]),cv::Range(0,Matrix3D.size[1]),cv::Range(k,k+1)};
	cv::Mat Matrix3DkOld=cv::Mat(Matrix3D,arrayOfRanges);//layer1Temp.copyTo(layer1);


	cv::MatConstIterator_<double> itMat3Dk=Matrix3Dk.begin<double>();
	cv::MatConstIterator_<double>  itMat3DkEnd=Matrix3Dk.end<double>();

	cv::MatIterator_<double> itMat3DkOld=Matrix3DkOld.begin<double>();
	cv::MatIterator_<double> itMat3DkOldEnd=Matrix3DkOld.end<double>();

	

	while ( itMat3Dk!= itMat3DkEnd and itMat3DkOld!= itMat3DkOldEnd) {
		(*itMat3DkOld)=(*itMat3Dk);
		itMat3DkOld++;itMat3Dk++;
	}

}

void printContentsOf3DCVMat(const cv::Mat &matrix,bool writeOnFile,std::string filename)
{
	if (writeOnFile==false)
	{
		std::cout<<"matrix of name : "<<filename<<" \n "<<std::endl;
 	if(matrix.dims==3)
 	{
 		cv::Mat matrixi;cv::Mat matrixij;
	for (int i=0;i<matrix.size[0];i++)
		{
			getRow3D(matrix,i,matrixi);
			for (int j=0;j<matrix.size[1];j++)
				{
					getRow2D(matrixi,j,matrixij);
					for (int k=0;k<matrix.size[2];k++)
						{
							std::cout<<"value of pixel"<<i <<" and "<<j<<" and "<<k<<" : "<< matrixij.at<double>(k)<<"  "<<std::endl;
						}	
				}
		}
	}
	if(matrix.dims==2)
 	{
 		cv::Mat matrixi;
	for (int i=0;i<matrix.size[0];i++)
		{
			getRow2D(matrix,i,matrixi);
			for (int j=0;j<matrix.size[1];j++)
				{

					std::cout<<"value of pixel"<<i <<" and "<<j<<" : "<< matrixi.at<double>(j)<<"  "<<std::endl;
				}
		}
	}
	if(matrix.dims==1)
 	{
	for (int i=0;i<matrix.size[0];i++)
		{
			// cv::Mat matrixi=getRow2D(matrix,i);
			// for (int j=0;j<matrix.size[1];j++)
				// {

					std::cout<<"value of pixel"<<i <<" : "<< matrix.at<double>(i)<<"  "<<std::endl;
				// }
		}
	}
	}
	else
	{
	// Declare what you need
		cv::FileStorage file(filename, cv::FileStorage::WRITE);

		file <<"the matrix"<< matrix;
	}
}

void castCVMatTovector_double(const cv::Mat &matrix,std::vector<double> &vector)
{
	// obtain iterator at initial position
	vector.resize(matrix.size[0]);
	cv::MatConstIterator_<double> itMat;
	std::vector<double>::iterator itVec=vector.begin();
	// obtain end position
	cv::MatConstIterator_<double> itMatend=matrix.end<double>();
	for ( itMat=matrix.begin<double>(); itMat!= itMatend; ++itMat) {
		(*itVec)=(*itMat);itVec+=1;
	}
}
void cast3DMatrixTo2DMatrix(const cv::Mat &matrix3D, cv::Mat &matrix2D)
{
	// We assume here that the 3D matrix have a size 1 on the third dimension
	// cv::MatConstIterator_<double> itMat;
	// cv::MatIterator_<double> itMat;
	// obtain end position
	
	int sizeLayer[2] = { matrix3D.size[0],matrix3D.size[1]};
	matrix2D=cv::Mat(2, sizeLayer, CV_64FC1, 0.0);

	cv::MatConstIterator_<double> itMat3D=matrix3D.begin<double>();
	cv::MatConstIterator_<double>  itMat3DEnd=matrix3D.end<double>();

	cv::MatIterator_<double> itMat2D=matrix2D.begin<double>();
	cv::MatIterator_<double> itMat2DEnd=matrix2D.end<double>();

	

	while ( itMat3D!= itMat3DEnd and itMat2D!= itMat2DEnd) {
		(*itMat2D)=(*itMat3D);
		itMat2D++;itMat3D++;
	}
}

void cast3DMatrixTo2DMatrixKeepingReference(const cv::Mat &matrix3D, cv::Mat &matrix2D)
{
	// Inputs:
	// the matrix3D has a size on the form {size[0],size[1],1}, and the matrix matrix2D is the same matrix, but without the last dimension

	int dims[] = { matrix3D.size[0], matrix3D.size[1]};

	matrix2D=cv::Mat(2,dims, CV_64FC1, matrix3D.data);

	// int dims[] = { Matrix2D.size[1]};

	// extractedMatrix=cv::Mat(1,dims, CV_64FC1, Matrix2D.data + Matrix2D.step[0] * numberRow);

}

void writeImageOnFloat(const cv::Mat &image,const std::string &name)
{
	cv::Mat imageCopy=image.clone();
	imageCopy.convertTo(imageCopy,CV_32FC1);
	iio_write_image_float(strdup(name.c_str()),(float *)imageCopy.data,imageCopy.size[1],imageCopy.size[0]);
}

void resizeWithShannonInterpolation(cv::Mat &image,cv::Mat &resizedImage,double zoom)
{
	// std::vector<cv::Mat> data_terms(1,m_g);
		image.convertTo(image,CV_32FC1);
		int size[2] = { int(floor(zoom*image.size[0])), int(floor(zoom*image.size[1]))};
		resizedImage=cv::Mat(2,size,CV_32FC1,0.0);//image1_resized.convertTo(image1_resized,CV_32FC1);
		image_zoom_2d((float*)image.data,(float*)resizedImage.data,image.size[1],image.size[0],resizedImage.size[1],resizedImage.size[0]);
		image.convertTo(image,CV_64FC1);
		resizedImage.convertTo(resizedImage,CV_64FC1);

		// printContentsOf3DCVMat(*m_image1,true,"image1");
		// printContentsOf3DCVMat(image1_resized,true,"image1_resized");
		// // writeImageOnFloat(*m_image1,"m_image1.tif");
		// // writeImageOnFloat(image1_resized,"image1_resized.tif");

		// image.convertTo(image,CV_64FC1);
		// resizedImage.convertTo(resizedImage,CV_64FC1);
		
		// printContentsOf3DCVMat(*m_image1,true,"image164");
		// printContentsOf3DCVMat(image1_resized,true,"image1_resized64");
}

std::string addSuffixFloatBeforeExtension(const std::string &nameInput,const float &suffix)
{
	int found=nameInput.find_first_of(".");
	std::string nameOutput=nameInput;
	nameOutput.insert(found,std::to_string(suffix));
	return nameOutput;
}
