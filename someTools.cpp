#include "someTools.h"

cv::Mat getLayer(cv::Mat Matrix3D,int layer_number)
// get a copy of the external layer of a 3D Matrix
{
	int size[2] = { (Matrix3D.size()).height, (Matrix3D.size()).width};
	cv::Mat layer(2, size, CV_64FC1, 0.0);
	for (int i=0;i<size[0];i++)
				{
					cv::Mat matrix3Di=getRow3D(Matrix3D,i);
					cv::Mat layeri=getRow2D(layer,i);
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



cv::Mat getLayer2D(const cv::Mat &matrix2D,int layer_number)
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

void testLayer2D()
{
	double m1[3][3] = { {10,1,2}, {3,4,5}, {6,7,8} };
	int size1[]={3,3};
	// // int sizeCh[3]= {50,50,50};
	cv::Mat M1(2,size1,CV_64FC1,m1);

	cv::Mat layer1=getLayer2D(M1,1);
	printContentsOf3DCVMat(M1,false,"M1");
	printContentsOf3DCVMat(layer1,false,"layer1");
	// cv::randu(layer1,-1.0,0.0);
	// printContentsOf3DCVMat(M1,true,"M1After");
	// printContentsOf3DCVMat(layer1,true,"layer1After");
}


cv::Mat getLayer3D(const cv::Mat &matrix3D,int layer_number)
// returns a reference on the 2D matrix matrix3D(.,.,layer_number)
{
	int sizeLayer[2] = { matrix3D.size[0],matrix3D.size[1]};
	cv::Mat layer=cv::Mat(2, sizeLayer, CV_64FC1, 0.0);

	cv::Mat matrix3Di;cv::Mat layeri;
	// MatConstIterator<double> it,it_end,// = M.end<double>();
	for (int i=0;i<matrix3D.size[0];i++)
	{
		matrix3Di=getRow3D(matrix3D,i);
		layeri=getRow2D(layer,i);
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
		getLayer2D(matrix3Di,layer_number).copyTo(layeri);
		// printContentsOf3DCVMat(layeri,false);
		// printContentsOf3DCVMat(layer,false);
	}
	return layer;

	// int sizeLayer[2] = { matrix3D.size[0],matrix3D.size[1]};
	// cv::Mat layer=cv::Mat(2, sizeLayer, CV_64FC1, 0.0);

	// uchar * matrix3Ddata=matrix3D.data;
	// // // // uchar layerdata[sizeof(matrix2Ddata)];
	// // // // layer.data;

	
	// int limit=matrix3D.step[0]*matrix3D.size[0]+ matrix3D.step[1]*matrix3D.size[1]+matrix3D.step[2]*layer_number;
	
	// int stepi= matrix3D.step[0];
	// int stepj= matrix3D.step[1];

	// int stepLayer=0;
	// // // memcpy(layer.data+0,matrix3Ddata,sizeof(double));

	// for(int i=matrix3D.step[2]*layer_number;i < limit;i+=stepi)
	// {
	// 	for(int j=matrix3D.step[2]*layer_number+i*matrix3D.step[0];j < limit;j+=stepj)
	// 	{
	// 		*(layer.data+stepLayer)=*(matrix3Ddata+k);
	// 	// layerdata
	// 	// layer.data=matrix3Ddata+k;
	// 	// stepLayer++;
	// 	// memcpy(layer.data+stepLayer,matrix3Ddata+k,sizeof(double));
	// 	// memcpy(layer.data+stepLayer*matrix3D.step[1],matrix3Ddata+k,sizeof(double));

	// 	stepLayer++;
	// }
	// return cv::Mat(2,sizeLayer,CV_64FC1,layer.data);
}

void testLayer3D()
{
	double m[3][3][3] = { {{0,1,2}, {3,4,5}, {6,7,8}}, {{10,11,12}, {13,14,15}, {16,17,18}},{{20,21,22}, {23,24,25}, {26,27,28}} };
	// // double m[3][3][3] = { {{0,0.5,1.0}, {1.0,1.5,2.0}, {2.0,2.5,3.0}}, {{0,0.5,1.0}, {1.0,1.5,2.0}, {2.0,2.5,3.0}} , {{0,0.5,1.0}, {1.0,1.5,2.0}, {2.0,2.5,3.0}} };
	int size[]={3,3,3};
	// // int sizeCh[3]= {50,50,50};
	cv::Mat M(3,size,CV_64FC1,m);
	cv::Mat layer=getLayer3D(M,0);
	printContentsOf3DCVMat(M,false,"M.txt");
	printContentsOf3DCVMat(layer,false,"layer.txt");
	// cv::randu(layer,-1.0,0.0);
	// printContentsOf3DCVMat(M,true,"MAfter.txt");
	// printContentsOf3DCVMat(layer,true,"layerAfter.txt");
}



cv::Mat getRow4D(const cv::Mat &Matrix4D,int numberRow,bool newOne)
// get the  row numer numberRow from a 4D matrix
{

	int dims[] = { Matrix4D.size[1], Matrix4D.size[2],Matrix4D.size[3]};
	if (numberRow > Matrix4D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	cv::Mat extractedMatrix(3,dims, CV_64FC1, Matrix4D.data + Matrix4D.step[0] * numberRow);
	return extractedMatrix;
}

cv::Mat getRow3D(const cv::Mat &Matrix3D,int numberRow)
// get the  row numer numberRow from a 4D matrix, not a copy but a reference to the real row
{

	int dims[] = { Matrix3D.size[1], Matrix3D.size[2]};
	if (numberRow > Matrix3D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	cv::Mat extractedMatrix(2,dims, CV_64FC1, Matrix3D.data + Matrix3D.step[0] * numberRow);
	return extractedMatrix;
}


cv::Mat getRow2D(const cv::Mat &Matrix2D,int numberRow)
// get the  row numer numberRow from a 4D matrix
{
	int dims[] = { Matrix2D.size[1]};
	if (numberRow > Matrix2D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	cv::Mat extractedMatrix(1,dims, CV_64FC1, Matrix2D.data + Matrix2D.step[0] * numberRow);
	return extractedMatrix;
}

void printContentsOf3DCVMat(const cv::Mat matrix,bool writeOnFile,std::string filename)
{
	if (writeOnFile==false)
	{
		std::cout<<"matrix of name : "<<filename<<" \n "<<std::endl;
 	if(matrix.dims==3)
 	{
	for (int i=0;i<matrix.size[0];i++)
		{
			cv::Mat matrixi=getRow3D(matrix,i);
			for (int j=0;j<matrix.size[1];j++)
				{
					cv::Mat matrixij=getRow2D(matrixi,j);
					for (int k=0;k<matrix.size[2];k++)
						{
							std::cout<<"value of pixel"<<i <<" and "<<j<<" and "<<k<<" : "<< matrixij.at<double>(k)<<"  "<<std::endl;
						}	
				}
		}
	}
	if(matrix.dims==2)
 	{
	for (int i=0;i<matrix.size[0];i++)
		{
			cv::Mat matrixi=getRow2D(matrix,i);
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