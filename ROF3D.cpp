#include "ROF3D.h"

// int m_nbMaxThreads=m_nbMaxThreads;

void ROF3D::testLab()
// just to do some tests for debugging
{
	int size[3] = { m_y_size,m_x_size,m_t_size};
	cv::Mat input=cv::Mat(3, size, CV_64FC1, 5.0);
	cv::Mat output=cv::Mat(3,size,CV_64FC1, 0.0);
	proxTVl(input,output);
	printContentsOf3DCVMat(output,true,"output");
	// double costArgmin=computeCostForArgumentTVh(m_x3Current,output);
	// testMinimialityOfSolutionTVL(input,output,25,0.00001);
	// throw std::invalid_argument( "testing the algorithm" );

}
void ROF3D::testContraintOnSolution(const cv::Mat &argminToTest)
// test if argminToTest verify the following contraint of the solutino 0<u(i,j,k)<1 u(i,j,0)=1 , u(i,j,m_t_size)=0 and u(i,j,k)>u(i,j,k+1)
{
	bool succes=true;
	cv::Mat argminToTesti;
	cv::Mat argminToTestij;

	std::cout<<"pixels for which the contraints are not verified"<<std::endl;
	for (int i=0;i<m_y_size;i++)
	{
		getRow3D(argminToTest,i,argminToTesti);
		for (int j=0;j<m_x_size;j++)
		{
			double * argminToTestij=argminToTesti.ptr<double>(j);
			if (argminToTestij[0]!=1.0 )
				{
					std::cout<<"value of pixel (i,j,k) =  ("<<i<<","<<j<<",0)"<<"equal to :"<<argminToTestij[0]<<std::endl;
					// throw std::invalid_argument( "u(i,j,0)!=1" );
				}
			if (argminToTestij[m_t_size-1]!=0.0 )
				{
					std::cout<<"value of pixel (i,j,k) =  ("<<i<<","<<j<<","<<m_t_size-1<<"equal to :"<<argminToTestij[m_t_size-1]<<std::endl;				
					// throw std::invalid_argument( "u(i,j,m_t_size-1)!=0" );
				}
			for (int k=1;k<m_t_size-1;k++)
			{
				// std::cout<<"value of pixel (i,j,k) =  ("<<i<<","<<j<<","<<k<<" ) equal to :"<<argminToTestij[k]<<std::endl;
				if(argminToTestij[k]<argminToTestij[k+1])
				{
					// for (int h=0;h<m_t_size;h++)
					// {
						std::cout<<"value of pixel (i,j,k) =  ("<<i<<","<<j<<","<<k<<" ) equal to :"<<argminToTestij[k]<<std::endl;
						std::cout<<"value of pixel (i,j,k+1) =  ("<<i<<","<<j<<","<<k+1<<" ) equal to :"<<argminToTestij[k+1]<<std::endl;
					// }
					// throw std::invalid_argument( "u(i,j,k)<u(i,j,k+1)" );
				}
			}
		}
	}
	std::cout<<" result of the test : "<<succes<<std::endl;
}

ROF3D::ROF3D(const cv::Mat & data_term,int Niter,const std::string &path_to_disparity,size_t nbMaxThreads,double precision) : m_nbMaxThreads(nbMaxThreads),m_precision(precision)
//this function resolve argmin_{v}( Sigma g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+Sigma |v(i,j+1,k)-v(i,j,k)|+Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma |v(i,j,k)-m_f(i,j,k)|^2) with v(i,j,k)
// on R
{
	
	data_term.copyTo(m_g);
	m_y_size=m_g.size[0];
	m_x_size=m_g.size[1];
	m_t_size=m_g.size[2]+1;// the cost must have a size smaller than 1 dimension for the last extension
	m_tau=0.5;
	m_lambda=1;
	initf();
	int size[3] = { m_y_size,m_x_size,m_t_size};
	m_x1Current=cv::Mat(3, size, CV_64FC1, 0.0);m_x1Previous=cv::Mat(3, size, CV_64FC1, 0.0);m_x1Bar=cv::Mat(3, size, CV_64FC1, 0.0);
	m_x2Current=cv::Mat(3, size, CV_64FC1, 0.0);m_x2Previous=cv::Mat(3, size, CV_64FC1, 0.0);m_x2Bar=cv::Mat(3, size, CV_64FC1, 0.0);
	m_x3Current=cv::Mat(3, size, CV_64FC1, 0.0);

	// iterate_algorithm();
	m_t_current=1;

	m_Niter=Niter;
	m_iteration=0;
	m_path_to_disparity=path_to_disparity;

	// cv::Mat input(3, size, CV_64FC1, 5.0);
	// cv::Mat output=cv::Mat(3,size,CV_64FC1, 0.0);
	// proxTVhOnTau(input,output);
	// printContentsOf3DCVMat(output,true,"output");
	// double costArgmin=computeCostForArgumentTVh(m_x3Current,output);
	// testMinimialityOfSolutionTVH(input,output,25,10);
	// proxTVl(m_f,output);
	// testMinimialityOfSolutionTVL(m_f,output,25,0.01);
	// proxTVvOnTau(m_f,output);
	// testMinimialityOfSolutionTVV(m_f,output,25,0.0001);
	// testLab();
	// throw std::invalid_argument( "testing the algorithm" );
	// printContentsOf3DCVMat(data_term,true,"data_termj");
	launch();
	computeMinSumTV();
	computeDisparity();
	// testContraintOnSolution(m_u);

	// proxTVhOnTau(m_f,output);
	// testMinimialityOfSolutionTVH(m_f,output,25,0.0001);
	// std::cout<<computeCostForArgumentTVl(m_tau,output);
	// m_x3Current=cv::Mat(3, size, CV_64FC1, 0.0);m_x3Previous=cv::Mat(3, size, CV_64FC1, 0.0);
	// MatchingAlgorithm::printContentsOf3DCVMat(MatchingAlgorithm::getLayer(m_f,0),false);
	// MatchingAlgorithm::printContentsOf3DCVMat(MatchingAlgorithm::getLayer(m_f,m_t_size-1),false);
	// MatchingAlgorithm::printContentsOf3DCVMat(MatchingAlgorithm::getLayer(m_f,3),false);
	// MatchingAlgorithm::getLayer(m_f,m_t_size-1)=cv::Mat(3, size, CV_64FC1, 0.0);
	
}


void ROF3D::testMinimalityOfSolution(int numberOfTests,double margin)
// test if ROF3D could computes the solution of the problem argmin(Sigma |v(i+1,j,k)-v(i,j,k)|+ Sigma |v(i,j+1,k)-v(i,j,k)|+ Sigma |v(i,j,k+1)-v(i,j,k)|*m_g(i,j,k) +Sigma (v(i,j,k)-m_f(i,j,k)))
{
	std::cout<<"test if ROF3D could computes the solution of the problem argmin(Sigma |v(i+1,j,k)-v(i,j,k)|+ Sigma |v(i,j+1,k)-v(i,j,k)|+ Sigma |v(i,j,k+1)-v(i,j,k)|*m_g(i,j,k) +Sigma (v(i,j,k)-m_f(i,j,k)))"<< std::endl;
	std::cout<<"number of tests"<<numberOfTests<<"margin"<<margin<<std::endl;

	double costArgmin=computeCostPrimal(m_v);
	// printContentsOf3DCVMat(argmin,true,"argmin.txt");
	cv::Mat argument;m_v.copyTo(argument);
	bool succes=true;
	for (int i=0;i<numberOfTests;i++)
	{
		cv::randu(argument,-margin,margin);
		argument=argument+m_v;
		double costArgument=computeCostPrimal(argument);
		std::cout<<"cost of the minimum : "<<costArgmin<<" and cost of a neighbor argument number : "<<i<<" :"<<costArgument<<" and difference :"<<costArgument-costArgmin<<std::endl; 
		succes=bool(costArgument-costArgmin>=0);
		// std::ostringstream ss;
		// ss << i;
		// std::string stringi=std::to_string(i);
		// printContentsOf3DCVMat(argument,true,"argument_"+ss.str()+".txt");
		// std::cout<<" the argument is ( "<<"the matrix"<< argument<<",";
		// std::cout<<" ) \n";
	}
	if (succes==true)
		{
			std::cout<<"\n \n for the "<<numberOfTests<<" generated arguments the minimizer succeded to minimize the cost function"<<std::endl;
		}
}
double ROF3D::computeCostPrimal(const cv::Mat &argument)
 //"argument" should de declared and initialized, this function compute the value of the cost function (the function to minimize):
 // "argument" is the primal variable of the primal problem
 // Sigma |v(i+1,j,k)-v(i,j,k)|+ Sigma |v(i,j+1,k)-v(i,j,k)|+ Sigma |v(i,j,k+1)-v(i,j,k)|*m_g(i,j,k) +m_lambda/2 Sigma (v(i,j,k)-m_f(i,j,k))

{
	double result=0;
 	int sizeArgumenty=argument.size[0];
	int sizeArgumentx=argument.size[1];
	int sizeArgumentt=argument.size[2];
	cv::Mat argumenti;cv::Mat m_gi;
	cv::Mat argumentip1;
	cv::Mat argumentij;
	cv::Mat m_gij;
	cv::Mat argumentijp1;
	cv::Mat argumentip1j;

	//computing weighted TVL
	for (int i=0;i<sizeArgumenty;i++)
	{
		getRow3D(argument,i,argumenti);
		getRow3D(m_g,i,m_gi);
		for (int j=0;j<sizeArgumentx;j++)
		{
			getRow2D(argumenti,j,argumentij);
			getRow2D(m_gi,j,m_gij);
			// for (int k=0;k<sizeArgumentt;k++)
			// {
			// 	result+=0.5*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			// }
			for (int k=0;k<sizeArgumentt-1;k++)
			{
				result+=std::abs(argumentij.at<double>(k+1)-argumentij.at<double>(k))*m_gij.at<double>(k);
			}
		}
	}
	

	//computing TVH
	for (int i=0;i<sizeArgumenty;i++)
			{
				getRow3D(argument,i,argumenti);
			for (int j=0;j<sizeArgumentx-1;j++)
				{
				getRow2D(argumenti,j,argumentij);
				getRow2D(argumenti,j+1,argumentijp1);
				for (int k=0;k<sizeArgumentt;k++)
				{
			// for (int i=0;i<sizeArgumenty;i++)
			// {
			// 	result+=0.5*m_tau*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			// }
			
				result+=std::abs(argumentijp1.at<double>(k)-argumentij.at<double>(k));
				}
		}
	}
	//computing TVV
	for (int i=0;i<sizeArgumenty-1;i++)
		{
			getRow3D(argument,i,argumenti);
			getRow3D(argument,i+1,argumentip1);
			// for (int j=0;j<sizeArgumentx;j++)
			// {
			// 	result+=0.5*m_tau*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			// }
		for (int j=0;j<sizeArgumentx;j++)
			{
			getRow2D(argumenti,j,argumentij);
			getRow2D(argumentip1,j,argumentip1j);
			for (int k=0;k<sizeArgumentt;k++)
				{
				result+=std::abs(argumentip1j.at<double>(k)-argumentij.at<double>(k));
				}
			}
		}
	double resultQuadraticTerm=0;
	// for (int i=0;i<sizeArgumenty;i++)
	// 	{
	// 		cv::Mat argumenti=getRow3D(argument,i);
	// 		cv::Mat m_fi=getRow3D(m_f,i);
	// 		for (int j=0;j<sizeArgumentx;j++)
	// 		{
	// 			cv::Mat argumentij=getRow2D(argumenti,j);
	// 			cv::Mat m_fij=getRow2D(m_fi,j);
	// 			for (int k=0;k<sizeArgumentt;k++)
	// 			{
	// 				resultQuadraticTerm+=pow(argumentij.at<double>(k)-m_fij.at<double>(k),2);
	// 			}
	// 		}
	// 	}
	cv::Mat QuadraticTerm=argument-m_f;
	resultQuadraticTerm=cv::sum((QuadraticTerm.mul(QuadraticTerm)))[0];
	resultQuadraticTerm=resultQuadraticTerm*m_lambda*0.5;
	result+=resultQuadraticTerm;

	return result;
}


double ROF3D::computeCostDual(const cv::Mat &x1,const cv::Mat &x2,const cv::Mat &x3)
// compute the dual cost should always be smaller than the primal cost
// the expresion of the dual is the following:
// -TVHStar(x1)-TVVStar(x2)-TVLStar(x3)+<x1+x2+x3/m_f>-1/(2lambda)*||x1+x2+x3||^2
{
	double result=0;
	result-=computeTVHStar(x1);
	if (result!=-INFINITY) result-=computeTVVStar(x2);
	if (result!=-INFINITY) result-=computeTVLStar(x3);
	if (result!=-INFINITY) 
	{
		cv::Mat sumXi=x1+x2+x3;
		result+=cv::sum(sumXi.mul(m_f))[0];
		result-=(cv::sum((sumXi.mul(sumXi)))[0])/(2.0*m_lambda);
	}
	return result;
}


double ROF3D::computeTVHStar(const cv::Mat & argument)
//compute the conjugate of the  TVH= Sigma |v(i,j+1,k)-v(i,j,k)|
//The argument is an cv::Mat object of dimension 3 and of type double
// the output is equal to infinity or 0
{
	double result=0;
	int size[]={argument.size[0],argument.size[1],argument.size[2]};
	cv::Mat argumenti;
	cv::Mat argumentipk;
	for (int i=0;i<size[0];i++)
	{
		getRow3D(argument,i,argumenti);
		// printContentsOf3DCVMat(argumenti,true,"argumentForDUAL");
		for (int k=0;k<size[2];k++)
		{
			getLayer2D(argumenti,k,argumentipk);
			result=computeTV1DStar(argumentipk);
			if (result==INFINITY) return result;
		}
	}
	return result;
}


double ROF3D::computeTVVStar(const cv::Mat & argument)
//compute the conjugate of t TVV=Sigma |v(i+1,j,k)-v(i,j,k)|.
//The argument is an cv::Mat object of dimension 3 and of type double
{
	double result=0;
	int size[]={argument.size[0],argument.size[1],argument.size[2]};
	cv::Mat argumentppk;
	cv::Mat argumentpjk;
	for (int k=0;k<size[2];k++)
	{
		getLayer3D(argument,k,argumentppk);
		for (int j=0;j<size[1];j++)
		{
			getLayer2D(argumentppk,j,argumentpjk);
			computeTV1DStar(argumentpjk);
			if (result==INFINITY) return result;
		}
	}
	return result;
}


double ROF3D::computeTVLStar(const cv::Mat & argument)
//compute the conjugate of  TVL=Sigma g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|.
//The argument is an cv::Mat object of dimension 3 and of type double
// m_g is normaly computed at the initialisation step
{
	double result=0;
	int size[]={argument.size[0],argument.size[1],argument.size[2]};
	cv::Mat argumenti;cv::Mat gi;
	cv::Mat argumentij;cv::Mat gij;
	for (int i=0;i<size[0];i++)
	{
		getRow3D(argument,i,argumenti);
		getRow3D(m_g,i,gi);
		for (int j=0;j<size[1];j++)
		{
			getRow2D(argumenti,j,argumentij);
			getRow2D(gi,j,gij);
			computeTV1DStarWeighted(argumentij,gij);
			if (result==INFINITY) return result;
			// if (infinityOrNot==true) break;
		}
	}
	return result;
}

double ROF3D::computeTV1DStar(const cv::Mat & argument)
//compute the conjugate of  TV1D. (with TV1D=Sigma_{1 <=j <= N-1} |u_{j+1}-u_{j}|)
//The argument is an cv::Mat object of dimension 1 and of type double
// the result is equal to zero only if |Sigma_{1<=k<=h} argument[k]| <=1 for all possible h and  Sigma_{k} argument[k]=0 (smaller then the term m_precision taking into account the numerical inmm_precision)
{
	// infinityOrNot=false;
	double result=0;
	double sum=0;
	// printContentsOf3DCVMat(argument,true,"argumentForDUAL");
	for (int h=0;h<argument.size[0]-1;h++)
	{
		sum+=argument.at<double>(h);
		if (std::abs(sum)>1+m_precision)
		{
			// infinityOrNot=true;
			result=INFINITY;
			return result;
			// break;
		}
	}
	sum+=argument.at<double>(argument.size[0]-1);
	if (std::abs(sum)>m_precision)
	{
		// infinityOrNot=true;
		result=INFINITY;
	}
	return result;
}

double ROF3D::computeTV1DStarWeighted(const cv::Mat & argument,const cv::Mat weight)
//compute the conjugate of the weighted TV1D.
//The weighted TV1D is equal to: Sigma_{i} |argument_{i+1}-argument_{i}|*weight_{i}
// the cv::Mat object weight is smaller by 1 than the object argument and all its terms are positive
// argument and  weighted are cv::Mat object of dimension 1 and of type double

// the result is equal to zero only if |Sigma_{1<=k<=h} argument[k]| <=weight[k] for all possible h and  Sigma_{k} argument[k]=0 (smaller then the term m_precision taking into account the numerical inmm_precision)

{
	// infinityOrNot=false;
	double result=0;
	double sum=0;
	for (int h=0;h<argument.size[0]-1;h++)
	{
		sum+=argument.at<double>(h);
		if (std::abs(sum)>weight.at<double>(h)+m_precision)
		{
			// infinityOrNot=true;
			result=INFINITY;
			return result;
			// break;
		}
	}
	sum+=argument.at<double>(argument.size[0]-1);
	if (std::abs(sum)>m_precision)
	{
		// infinityOrNot=true;
		result=INFINITY;
	}
	return result;

}
// void ROF3D::testMinimalityOfSolution(const cv::Mat &argmin,int numberOfTests,double margin);
// {
	

// }




void ROF3D::launch()
//after the methodes init has been launched
{
	std::cout<<"solving ROF3D problem "<<std::endl;
	while( m_iteration < m_Niter)// and ( m_gap >= m_factor*m_gapInit ) )
	{
		iterate_algorithm();
		// if(m_iteration%10==0) disparity_estimation();
		computeMinSumTV();//m_v=m_f-(1/m_lambda)*(m_x1Current+m_x2Current+m_x3Current);
		computeDisparity();
		double primalCost=computeCostPrimal(m_v);
		double dualCost=computeCostDual(m_x1Current,m_x2Current,m_x3Current);
		std::cout<<" the cost of the primal is :"<<primalCost<<" and the cost of the dual is : "<<dualCost<<std::endl;
		std::cout<<"the dual gap is : "<<primalCost-dualCost;
	}
}


void ROF3D::iterate_algorithm()
{
	double t_next=(1+sqrt(1+4*pow(m_t_current,2)))/2;
	m_x1Bar=m_x1Current+((m_t_current-1)/t_next)*(m_x1Current- m_x1Previous);
	m_x2Bar=m_x2Current+((m_t_current-1)/t_next)*(m_x2Current- m_x2Previous);
	proxTVlStar(m_lambda*m_f-(m_x1Bar+m_x2Bar),m_x3Current);
	m_x1Current.copyTo(m_x1Previous);
	m_x2Current.copyTo(m_x2Previous);
	proxtauTVhStar(m_x1Bar- m_tau*(m_x1Bar+m_x2Bar+m_x3Current- m_lambda*m_f),m_x1Current);
	proxtauTVvStar(m_x2Bar- m_tau*(m_x1Bar+m_x2Bar+m_x3Current- m_lambda*m_f),m_x2Current);

	m_iteration+=1;
	std::cout<<"\n iteration number : "<<m_iteration<<" performed "<<std::endl;

	// std::cout<<"iteration number : "<<m_iteration<<" performed "<<" and gap equal to "<<m_gap<<std::endl;

}

void ROF3D::computeMinSumTV()
{
	// m_x1Current.mul(m_f)-TVh
	m_v=m_f-(1/m_lambda)*(m_x1Current+m_x2Current+m_x3Current);
	// printContentsOf3DCVMat(getLayer3D(m_v,0),false);std::cout<<"over \n";throw std::invalid_argument( "testing the algorithm" );
	// printContentsOf3DCVMat(m_v,true,"m_v");
	// m_u=convertTo((m_v < 0.0),CV_64FC1);
	// cv::Mat doubleV0;
	cv::Mat m_u_bool=(m_v > 0.0);
    m_u_bool.convertTo(m_u, CV_64FC1);
    m_u=m_u/255.0;
    // printContentsOf3DCVMat(getLayer3D(m_u,0),false);
    // printContentsOf3DCVMat(getLayer3D(m_u,m_t_size-1),false);
    // testContraintOnSolution(m_u);

    // cv::Mat v0 = (divv < 0.0);

	// cv::Mat doubleV0;
    // v0.convertTo(doubleV0, CV_64FC1);
    // printContentsOf3DCVMat(m_f,false);
    // printContentsOf3DCVMat(m_v,false);

}

void ROF3D::computeDisparity()
{
	m_disparity=cv::Mat(m_u.size[0],m_u.size[1],CV_64FC1,0.0);

	// cv::Mat thresholded = (m_u > 0.5);
	// int z=thresholded.size[2];

	// int size[3] = { m_y_size, m_x_size, m_t_size };
	// cv::Mat doubleThresholded=cv::Mat(3, size, CV_64FC1, 0.0);
	// thresholded.copyTo(doubleThresholded);
 //    thresholded.convertTo(doubleThresholded, CV_64FC1);
 //    // printContentsOf3DCVMat(doubleThresholded,true);
	double zoomFactor=255.0/(double(m_u.size[2]));
	// // cv::Mat thresholdedDouble;
 //    // thresholded.convertTo(thresholdedDouble, CV_64FC1);
 //    // int size[4]= {3,v.size[0],v.size[1],v.size[2]};
	// 	// cv::Mat delta(4,size,CV_64FC1,0.0);
	for (int i = 0; i < m_u.size[0]; i++)
	{
		cv::Mat m_ui = MatchingAlgorithm::getRow3D(m_u,i);
		double * disparityi = m_disparity.ptr<double>(i);
		for (int j = 0; j < m_u.size[1]; j++)
		{
			double * m_uij = m_ui.ptr<double>(j);
			disparityi[j]=0;
			for (int k=0;k< m_u.size[2];k++)
			{
				disparityi[j]+=m_uij[k];
			}
			disparityi[j]*=zoomFactor;
		}
	}

    imwrite(m_path_to_disparity,m_disparity);
}

cv::Mat ROF3D::getSolution()
{
	return m_u;

}

void ROF3D::proxTVlStar(const cv::Mat &input,cv::Mat &output)
{
	proxTVl(input,output);
	output=input-output;
}
void ROF3D::proxtauTVhStar(const cv::Mat &input,cv::Mat &output)
{
	proxTVhOnTau((1/m_tau)*input,output);
	output=input-m_tau*output;
}
void ROF3D::proxtauTVvStar(const cv::Mat &input,cv::Mat &output)
{
	proxTVvOnTau((1/m_tau)*input,output);
	output=input-m_tau*output;
}


void ROF3D::proxTVl(const cv::Mat &input,cv::Mat &output)

// this function  computes prox(TVl(g))(input): it resolves the problem argmin( Sigma m_g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R, which is the element "output"
// description: it resolves the problem argmin( Sigma m_g(i,j,k)*|v_{i,j}(k+1)-v_{i,j}(k)|+m_tau/2*Sigma|v_{i,j}(k)-input(i,j,k)|)
{	
	int sizeInputy=input.size[0];
	int sizeBlock=1;
	int N=int(sizeInputy/sizeBlock);
	ThreadPool threadPool(m_nbMaxThreads);
	for (int i=0;i < N;i++)
	{
		// beginIncluded=i*sizeBlock;
		// endExcluded=(i+1)*sizeBlock;
		threadPool.enqueue(&ROF3D::proxTVLijExternMultiple,*this,input,output,sizeBlock*i,sizeBlock*(i+1));

	}
	// beginIncluded=N*sizeBlock;
	// endExcluded=sizeInputy;
	threadPool.enqueue(&ROF3D::proxTVLijExternMultiple,*this,input,output,sizeBlock*N,sizeInputy);// if sizeInputy=N*sizeBlock, nothing is done
}





void ROF3D::proxTVLij(const cv::Mat &inputi,cv::Mat &outputi,const cv::Mat &gi,int j)
{
	const double * inputij=inputi.ptr<double>(j);
	int sizeInputt=inputi.size[1];
	std::vector<double> inputijVect=std::vector<double>(inputij,inputij+sizeInputt);

	double * outputij=outputi.ptr<double>(j);

	const double * gij=gi.ptr<double>(j);
	std::vector<double> gijVect=std::vector<double>(gij,gij+sizeInputt-1);// We remove the last elements of gij because, we won't need it on the ROF computation

	ROF rof=ROF(1.0,inputijVect,gijVect);// this function resolves argmin_{u}( Sigma gij(k)|u(k+1)-u(k)|+1/2*Sigma|u(k)-inputij(k)|^2)
			//this function resolve argmin_{v(i,j,.)}( Sigma_{k} g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+1/2*Sigma_{k} |v(i,j,k)-l(i,j,k)|^2) with v(i,j,k) on R
	std::deque<double> outputijDeque=rof.getSolution(false); // printContentsOf3DCVMat(getRow2D(inputi,j),true,"fij.txt");
	
	for (int k=0;k<sizeInputt;k++)
		{
			outputij[k]=outputijDeque[k];
		}
}	

void ROF3D::proxTVLijExtern(const cv::Mat &inputi,cv::Mat &outputi,const cv::Mat &gi)
// looping of the previous function: "proxTVLij" ,it is still parralelizable
{
	int sizeInputx=inputi.size[0];
	for (int j=0;j<sizeInputx;j++)
		{
			proxTVLij(inputi,outputi,gi,j);
		}
}
void ROF3D::proxTVLijExternMultiple(const cv::Mat &input,cv::Mat &output,int beginIncluded,int endExcluded)
// looping of the previous function: "proxTVLijExtern" ,it is still parralelizable
{
	// int sizeInputy=input.size[0];
	cv::Mat inputi;
	cv::Mat gi;
	cv::Mat outputi;
	for (int i=beginIncluded;i<endExcluded;i++)
	{
		getRow3D(input,i,inputi);
		getRow3D(output,i,outputi);
		getRow3D(m_g,i,gi);
		proxTVLijExtern(inputi,outputi,gi);

	}
}
void ROF3D::proxTVvOnTau(const cv::Mat &input,cv::Mat &output)

// // this function  computes prox(1/m_tau*TVv(input): it resolves the problem argmin( Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
// // the m_f is intialized before and refer to the object introduced to transform the problem of minimization to a ROF problem
	
	int sizeInputt=input.size[2];
	int sizeBlock=1;
	int N=int(sizeInputt/sizeBlock);
	ThreadPool threadPool(m_nbMaxThreads);
	for (int i=0;i < N;i++)
	{
		// beginIncluded=i*sizeBlock;
		// endExcluded=(i+1)*sizeBlock;
		threadPool.enqueue(&ROF3D::proxTVvOnTaupjkExternMultiple,*this,input,output,sizeBlock*i,sizeBlock*(i+1));
	}
	// beginIncluded=N*sizeBlock;
	// endExcluded=sizeInputy;
	threadPool.enqueue(&ROF3D::proxTVvOnTaupjkExternMultiple,*this,input,output,sizeBlock*N,sizeInputt);// if sizeInputy=N*sizeBlock, nothing is done
}



void ROF3D::proxTVvOnTaupjk(const cv::Mat &inputppk, cv::Mat &output,int j,int k)
//linked to the method "proxTVvOnTau", this function resolves the problem argmin( Sigma |v(i+1)-v(i)|+m_tau/2*Sigma|v(i)-input(i,j,k)|) with v(i) on R and fixed j and k
// arguments :
// -inputppk is the elements input(.,.,k) for k fixed 
// output is the solution of the problem
{

	cv::Mat inputpjk;
	getLayer2D(inputppk,j,inputpjk);
	std::vector<double> inputpjkVect;
	// inputpjk.copyTo(inputpjkVect);
	castCVMatTovector_double(inputpjk,inputpjkVect);
	// outputpjk=getLayer2D(outputppk,j);
	// double * inputij=inputi.ptr<double>(j);
	int sizeInputy=inputpjk.size[0];
	ROF rof=ROF(m_tau,inputpjkVect);// this function resolves argmin_{u}( Sigma gij(k)|u(k+1)-u(k)|+1/2*Sigma|u(k)-inputij(k)|^2)
			//this function resolve argmin_{v(i,j,.)}( Sigma_{k} g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+1/2*Sigma_{k} |v(i,j,k)-l(i,j,k)|^2) with v(i,j,k) on R

	std::deque<double> outputpjkDeque=rof.getSolution(false); // printContentsOf3DCVMat(getRow2D(inputi,j),true,"fij.txt");
	
	for (int i=0;i<sizeInputy;i++)
		{
			output.at<double>(cv::Vec<int,3>(i,j,k))=outputpjkDeque[i];
		}
	// writing the result on the associated part of the memory
}

void ROF3D::proxTVvOnTaupjkExtern(const cv::Mat &inputppk, cv::Mat &output,int k)
{
	int sizeInputx=inputppk.size[1];
	for (int j=0;j<sizeInputx;j++)
		{
			proxTVvOnTaupjk(inputppk,output,j,k);
		}
}

void ROF3D::proxTVvOnTaupjkExternMultiple(const cv::Mat &input,cv::Mat &output,int beginIncluded,int endExcluded)

// // this function  computes prox(1/m_tau*TVv(input): it resolves the problem argmin( Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
// // the m_f is intialized before and refer to the object introduced to transform the problem of minimization to a ROF problem
	
	
	// int sizeInputt=input.size[2];
	cv::Mat inputppk;
	for (int k=beginIncluded;k<endExcluded;k++)
	{
		getLayer3D(input,k,inputppk);
		proxTVvOnTaupjkExtern(inputppk,output,k);
	}
}


void ROF3D::proxTVhOnTau(const cv::Mat &input,cv::Mat &output)

// // this function  computes prox(1/m_tau*TVv(input): it resolves the problem argmin( Sigma |v(i,j+1,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
// // the m_f is intialized before and refer to the object introduced to transform the problem of minimization to a ROF problem
	
	
	int sizeInputy=input.size[0];
	int sizeBlock=1;
	int N=int(sizeInputy/sizeBlock);
	ThreadPool threadPool(m_nbMaxThreads);
	for (int i=0;i < N;i++)
	{
		// beginIncluded=i*sizeBlock;
		// endExcluded=(i+1)*sizeBlock;
		threadPool.enqueue(&ROF3D::proxTVhOnTauipkExternMultiple,*this,input,output,sizeBlock*i,sizeBlock*(i+1));
	}
	// beginIncluded=N*sizeBlock;
	// endExcluded=sizeInputy;
	threadPool.enqueue(&ROF3D::proxTVhOnTauipkExternMultiple,*this,input,output,sizeBlock*N,sizeInputy);// if sizeInputy=N*sizeBlock, nothing is done

}



void ROF3D::proxTVhOnTauipk(const cv::Mat &inputi, cv::Mat &outputi,int k)
{
	cv::Mat inputipk;
	getLayer2D(inputi,k,inputipk);
	std::vector<double> inputipkVect;
	// inputipk.copyTo(inputipkVect);
	castCVMatTovector_double(inputipk,inputipkVect);


	int sizeInputx=outputi.size[0];
	ROF rof=ROF(m_tau,inputipkVect);// this function resolves argmin_{u}( Sigma gij(k)|u(k+1)-u(k)|+1/2*Sigma|u(k)-inputij(k)|^2)
	
	//this function resolve argmin_{v(i,j,.)}( Sigma_{k} g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+1/2*Sigma_{k} |v(i,j,k)-l(i,j,k)|^2) with v(i,j,k) on R

	std::deque<double> outputipkDeque=rof.getSolution(false); // printContentsOf3DCVMat(getRow2D(inputi,j),true,"fij.txt");
	
	for (int j=0;j<sizeInputx;j++)
		{
			outputi.at<double>(cv::Vec<int,2>(j,k))=outputipkDeque[j];
		}
		// writing the result on the associated part of the memory
}

void ROF3D::proxTVhOnTauipkExtern(const cv::Mat &inputi, cv::Mat &outputi)
{
	int sizeInputt=inputi.size[1];
	for (int k=0;k<sizeInputt;k++)
		{
			proxTVhOnTauipk(inputi,outputi,k);
		}
}

void ROF3D::proxTVhOnTauipkExternMultiple(const cv::Mat &input,cv::Mat &output,int beginIncluded,int endExcluded)

// // this function  computes prox(1/m_tau*TVv(input): it resolves the problem argmin( Sigma |v(i,j+1,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
// // the m_f is intialized before and refer to the object introduced to transform the problem of minimization to a ROF problem
	
	
	// int sizeInputy=input.size[0];
	cv::Mat inputi;
	cv::Mat outputi;
	for (int i=beginIncluded;i<endExcluded;i++)
	{
		getRow3D(input,i,inputi);
		getRow3D(output,i,outputi);
		proxTVhOnTauipkExtern(inputi,outputi);
	}
}

void ROF3D::testMinimialityOfSolutionTVL(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin)
// test if proxTVl could computes prox(1/m_tau*TVl(g))(input): it resolves the problem argmin( Sigma m_g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
	std::cout<<"test if proxTVl could computes prox(1/m_tau*TVl(g))(input): it resolves the problem argmin( Sigma m_g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R"<< std::endl;
	double costArgmin=computeCostForArgumentTVl(input,argmin);
	printContentsOf3DCVMat(argmin,true,"argmin.txt");
	cv::Mat argument;argmin.copyTo(argument);
	bool succes=true;
	for (int i=0;i<numberOfTests;i++)
	{
		cv::randu(argument,-margin,margin);
		argument=argument+argmin;
		double costArgument=computeCostForArgumentTVl(input,argument);
		std::cout<<"cost of the minimum : "<<costArgmin<<" and cost of a neighbor argument number : "<<i<<" :"<<costArgument<<" and difference :"<<costArgument-costArgmin<<std::endl; 
		succes=bool(costArgument-costArgmin>=0);
		std::ostringstream ss;
		ss << i;
		// std::string stringi=std::to_string(i);
		printContentsOf3DCVMat(argument,true,"argument_"+ss.str()+".txt");
		// std::cout<<" the argument is ( "<<"the matrix"<< argument<<",";
		// std::cout<<" ) \n";
	}
	if (succes==true)
		{
			std::cout<<"\n \n for the "<<numberOfTests<<" generated arguments the minimizer succeded to minimize the cost function"<<std::endl;
		}
}



 double ROF3D::computeCostForArgumentTVl(const cv::Mat &l,const cv::Mat &argument)
 //argument should de declared and initialized, this function compute the value of the minimized function of the proximal operator of 1/tau*TVL for the argument argument and the cost m_g:
 // it computes Sigma m_g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+tau/2*Sigma|v(i,j,k)-l(i,j,k)| with v the argument and the cost
 {
 	double result=0;
 	int sizeArgumenty=argument.size[0];
	int sizeArgumentx=argument.size[1];
	int sizeArgumentt=argument.size[2];
	for (int i=0;i<sizeArgumenty;i++)
	{
		for (int j=0;j<sizeArgumentx;j++)
		{
			for (int k=0;k<sizeArgumentt;k++)
			{
				result+=0.5*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			}
			for (int k=0;k<sizeArgumentt-1;k++)
			{
				result+=std::abs(argument.at<double>(i,j,k+1)-argument.at<double>(i,j,k))*m_g.at<double>(i,j,k);
			}
		}
	}
	return result;
 }

void ROF3D::testMinimialityOfSolutionTVV(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin)
// test if proxTVvOnTau could computes prox(1/m_tau*TVl(g))(input): it resolves the problem argmin( Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
	std::cout<<"test if proxTVvOnTau could computes prox(1/m_tau*TVl(g))(input): it resolves the problem argmin( Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R"<< std::endl;
	double costArgmin=computeCostForArgumentTVv(input,argmin);
	printContentsOf3DCVMat(argmin,true,"argmin.txt");
	cv::Mat argument;argmin.copyTo(argument);
	bool succes=true;
	for (int i=0;i<numberOfTests;i++)
	{
		cv::randu(argument,-margin,margin);
		argument=argument+argmin;
		double costArgument=computeCostForArgumentTVv(input,argument);
		std::cout<<"cost of the minimum : "<<costArgmin<<" and cost of a neighbor argument number : "<<i<<" :"<<costArgument<<" and difference :"<<costArgument-costArgmin<<std::endl; 
		succes=bool(costArgument-costArgmin>=0);
		std::ostringstream ss;
		ss << i;
		// std::string stringi=std::to_string(i);
		printContentsOf3DCVMat(argument,true,"argument_"+ss.str()+".txt");
		// std::cout<<" the argument is ( "<<"the matrix"<< argument<<",";
		// std::cout<<" ) \n";
	}
	if (succes==true)
		{
			std::cout<<"\n \n for the "<<numberOfTests<<" generated arguments the minimizer succeded to minimize the cost function"<<std::endl;
		}
}


 double ROF3D::computeCostForArgumentTVv(const cv::Mat &l,const cv::Mat &argument)
 //argument should de declared and initialized, this function compute the value of the minimized function of the proximal operator of 1/tau*TVL for the argument argument and the cost m_g:
 // it computes Sigma |v(i+1,j,k)-v(i,j,k)|+tau/2*Sigma|v(i,j,k)-l(i,j,k)| with v the argument and the cost
 {
 	double result=0;
 	int sizeArgumenty=argument.size[0];
	int sizeArgumentx=argument.size[1];
	int sizeArgumentt=argument.size[2];
	for (int k=0;k<sizeArgumenty;k++)
	{
		for (int j=0;j<sizeArgumentx;j++)
		{
			for (int i=0;i<sizeArgumenty;i++)
			{
				result+=0.5*m_tau*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			}
			for (int i=0;i<sizeArgumenty-1;i++)
			{
				result+=std::abs(argument.at<double>(i+1,j,k)-argument.at<double>(i,j,k));
			}
		}
	}
	return result;
 }

void ROF3D::testMinimialityOfSolutionTVH(const cv::Mat &input,const cv::Mat &argmin,int numberOfTests,double margin)
// test if proxTVhOnTau could computes prox(1/m_tau*TVh(g))(input): it resolves the problem argmin( Sigma |v(i,j+1,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
	std::cout<<"test if proxTVhOnTau could computes prox(1/m_tau*TVh(g))(input): it resolves the problem argmin( Sigma |v(i,j+1,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R"<< std::endl;

	double costArgmin=computeCostForArgumentTVh(input,argmin);
	printContentsOf3DCVMat(argmin,true,"argmin.txt");
	cv::Mat argument;argmin.copyTo(argument);
	bool succes=true;
	for (int i=0;i<numberOfTests;i++)
	{
		cv::randu(argument,-margin,margin);
		argument=argument+argmin;
		double costArgument=computeCostForArgumentTVh(input,argument);
		std::cout<<"cost of the minimum : "<<costArgmin<<" and cost of a neighbor argument number : "<<i<<" :"<<costArgument<<" and difference :"<<costArgument-costArgmin<<std::endl; 
		succes=bool(costArgument-costArgmin>=0);
		std::ostringstream ss;
		ss << i;
		// std::string stringi=std::to_string(i);
		printContentsOf3DCVMat(argument,true,"argument_"+ss.str()+".txt");
		// std::cout<<" the argument is ( "<<"the matrix"<< argument<<",";
		// std::cout<<" ) \n";
	}
	if (succes==true)
		{
			std::cout<<"\n \n for the "<<numberOfTests<<" generated arguments the minimizer succeded to minimize the cost function"<<std::endl;
		}
}

 double ROF3D::computeCostForArgumentTVh(const cv::Mat &l,const cv::Mat &argument)
 //argument should de declared and initialized, this function compute the value of the minimized function of the proximal operator of 1/tau*TVL for the argument argument and the cost m_g:
 // it computes Sigma |v(i,j+1,k)-v(i,j,k)|+tau/2*Sigma|v(i,j,k)-l(i,j,k)|^2 with v the argument and the cost
 {
 	double result=0;
 	int sizeArgumenty=argument.size[0];
	int sizeArgumentx=argument.size[1];
	int sizeArgumentt=argument.size[2];
	for (int k=0;k<sizeArgumenty;k++)
	{
		for (int i=0;i<sizeArgumenty;i++)
		{
			for (int j=0;j<sizeArgumentx;j++)
			{
				result+=0.5*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			}
			for (int j=0;j<sizeArgumentx-1;j++)
			{
				result+=std::abs(argument.at<double>(i,j+1,k)-argument.at<double>(i,j,k));
			}
		}
	}
	return result;
 }

cv::Mat ROF3D::getSolutionOfOriginalProblem()
// return not the solution of the 3D ROF Problem but the solution of the problem 
// argmin_{v}( Sigma g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+Sigma |v(i,j+1,k)-v(i,j,k)|+Sigma |v(i+1,j,k)-v(i,j,k)|) with u(i,j,k)  verifying the following contraints:
// 0<u(i,j,k)<1 u(i,j,0)=1 , u(i,j,m_t_size)=0 and u(i,j,k)>u(i,j,k+1)

{
	testContraintOnSolution(m_u);
	return m_u;
}









void ROF3D::initf(double delta)
{
	int size[3] = { m_y_size,m_x_size,m_t_size};
	m_f=cv::Mat(3, size, CV_64FC1, 0.0);
	cv::Mat fi;cv::Mat fij;
	for (int i = 0; i < m_y_size; i++)
	{	
		fi=MatchingAlgorithm::getRow3D(m_f,i);
  		for (int j = 0; j < m_x_size; j++)
    	{	
    		fij=MatchingAlgorithm::getRow2D(fi,j);
    		// double a=pow(10,3);
			fij.at<double>(0)=delta;
			fij.at<double>(m_t_size-1)=-delta;
		}
	}
}
