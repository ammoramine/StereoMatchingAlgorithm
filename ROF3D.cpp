#include "ROF3D.h"


void ROF3D::testLab()
// just to do some tests for debugging
{
	int size[3] = { m_y_size,m_x_size,m_t_size};
	cv::Mat input=cv::Mat(3, size, CV_64FC1, 5.0);
	cv::Mat output=cv::Mat(3,size,CV_64FC1, 0.0);
	proxTVl(input,output);
	printContentsOf3DCVMat(output,true,"output");
	// double costArgmin=computeCostForArgumentTVh(m_x3Current,output);
	testMinimialityOfSolutionTVL(input,output,25,0.00001);
	throw std::invalid_argument( "testing the algorithm" );

}

ROF3D::ROF3D(const cv::Mat & data_term,int Niter,const std::string &path_to_disparity)
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

	double costArgmin=computeTotalCost(m_v);
	// printContentsOf3DCVMat(argmin,true,"argmin.txt");
	cv::Mat argument;m_v.copyTo(argument);
	bool succes=true;
	for (int i=0;i<numberOfTests;i++)
	{
		cv::randu(argument,-margin,margin);
		argument=argument+m_v;
		double costArgument=computeTotalCost(argument);
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
double ROF3D::computeTotalCost(const cv::Mat argument)
 //argument should de declared and initialized, this function compute the value of the cost function (the function to minimize):

 // Sigma |v(i+1,j,k)-v(i,j,k)|+ Sigma |v(i,j+1,k)-v(i,j,k)|+ Sigma |v(i,j,k+1)-v(i,j,k)|*m_g(i,j,k) +Sigme (v(i,j,k)-m_f(i,j,k))

{
	double result=0;
 	int sizeArgumenty=argument.size[0];
	int sizeArgumentx=argument.size[1];
	int sizeArgumentt=argument.size[2];
	// cv::Mat argumenti;cv::Mat m_gi;
	// cv::Mat argumentij;cv::Mat m_gij;

	for (int i=0;i<sizeArgumenty;i++)
	{
		cv::Mat argumenti=getRow3D(argument,i);
		cv::Mat m_gi=getRow3D(m_g,i);
		for (int j=0;j<sizeArgumentx;j++)
		{
			cv::Mat argumentij=getRow2D(argumenti,j);
			cv::Mat m_gij=getRow2D(m_gi,j);
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
	// cv::Mat argumentk;
	// cv::Mat argumentij;cv::Mat m_gij;
	for (int i=0;i<sizeArgumenty;i++)
			{
				cv::Mat argumenti=getRow3D(argument,i);
			for (int j=0;j<sizeArgumentx-1;j++)
				{
				cv::Mat argumentij=getRow2D(argumenti,j);
				cv::Mat argumentijp1=getRow2D(argumenti,j+1);
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

	for (int i=0;i<sizeArgumenty;i++)
		{
			cv::Mat argumenti=getRow3D(argument,i);
			cv::Mat argumentip1=getRow3D(argument,i+1);
			// for (int j=0;j<sizeArgumentx;j++)
			// {
			// 	result+=0.5*m_tau*pow(argument.at<double>(i,j,k)-l.at<double>(i,j,k),2);
			// }
		for (int j=0;j<sizeArgumentx-1;j++)
			{
			cv::Mat argumentij=getRow2D(argumenti,j);
			cv::Mat argumentip1j=getRow2D(argumentip1,j);
			for (int k=0;k<sizeArgumentt;k++)
				{
				result+=std::abs(argumentip1j.at<double>(k)-argumentij.at<double>(k));
				}
			}
		}

	for (int i=0;i<sizeArgumenty;i++)
		{
			cv::Mat argumenti=getRow3D(argument,i);
			cv::Mat m_fi=getRow3D(m_f,i);
			for (int j=0;j<sizeArgumentx;j++)
			{
				cv::Mat argumentij=getRow2D(argumenti,j);
				cv::Mat m_fij=getRow2D(m_fi,j);
				for (int k=0;k<sizeArgumentt;k++)
				{
					result+=pow(argumentij.at<double>(k)-m_fij.at<double>(k),2);
				}
			}
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
		computeMinSumTV();
		computeDisparity();
		std::cout<<" the cost is :"<<computeTotalCost(m_v)<<std::endl;
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
	proxtauTVhStar(m_x2Bar- m_tau*(m_x1Bar+m_x2Bar+m_x3Current- m_lambda*m_f),m_x2Current);

	m_iteration+=1;
	std::cout<<"iteration number : "<<m_iteration<<" performed "<<std::endl;

	// std::cout<<"iteration number : "<<m_iteration<<" performed "<<" and gap equal to "<<m_gap<<std::endl;

}

void ROF3D::computeMinSumTV()
{
	// m_x1Current.mul(m_f)-TVh
	m_v=m_f-(1/m_lambda)*(m_x1Current+m_x2Current+m_x3Current);

	// m_u=convertTo((m_v < 0.0),CV_64FC1);
	// cv::Mat doubleV0;
	cv::Mat m_u_bool=(m_v < 0.0);
    m_u_bool.convertTo(m_u, CV_64FC1);
    // cv::Mat v0 = (divv < 0.0);

	// cv::Mat doubleV0;
    // v0.convertTo(doubleV0, CV_64FC1);
    // printContentsOf3DCVMat(m_f,false);
    // printContentsOf3DCVMat(m_v,false);
    // printContentsOf3DCVMat(m_u,false);

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
	double zoomFactor=1/(double(m_u.size[2]));
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
	proxTVhOnTau(1/m_tau*input,output);
	output=input-m_tau*output;
}
void ROF3D::proxtauTVvStar(const cv::Mat &input,cv::Mat &output)
{
	proxTVvOnTau(1/m_tau*input,output);
	output=input-m_tau*output;
}


void ROF3D::proxTVl(const cv::Mat &input,cv::Mat &output)

// this function  computes prox(1/m_tau*TVl(g))(input): it resolves the problem argmin( Sigma m_g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
	
	
	int sizeInputy=input.size[0];
	int sizeInputx=input.size[1];
	int sizeInputt=input.size[2];
	cv::Mat inputi;
	std::vector<double> inputijVect;
	cv::Mat gi;
	std::vector<double> gijVect;
	// std::vector<double> inputijVect;
	// int size[3] = { sizeInputy,sizeInputX,sizeInputt};
	// output=cv::Mat(3,size,CV_64FC1, 0.0);
	// cv::Mat output;
	cv::Mat outputi;
	std::deque<double> outputijDeque;

	for (int i=0;i<sizeInputy;i++)
	{
		inputi=getRow3D(input,i);
		outputi=getRow3D(output,i);
		gi=getRow3D(m_g,i);

		for (int j=0;j<sizeInputx;j++)
		{
			double * inputij=inputi.ptr<double>(j);
			inputijVect=std::vector<double>(inputij,inputij+sizeInputt);

			double * outputij=outputi.ptr<double>(j);

			double * gij=gi.ptr<double>(j);
			gijVect=std::vector<double>(gij,gij+sizeInputt-1);

			ROF rof=ROF(1.0,inputijVect,gijVect);// this function resolves argmin_{u}( Sigma gij(k)|u(k+1)-u(k)|+1/2*Sigma|u(k)-inputij(k)|^2)
			//this function resolve argmin_{v(i,j,.)}( Sigma_{k} g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+1/2*Sigma_{k} |v(i,j,k)-l(i,j,k)|^2) with v(i,j,k) on R
			outputijDeque=rof.getSolution(false);
			// printContentsOf3DCVMat(getRow2D(inputi,j),true,"fij.txt");
			for (int k=0;k<sizeInputt;k++)
			{
				outputij[k]=outputijDeque[k];
			}

			// ROF(m_m_tau,fijVect);
			// double bArray[] = {1,1,1,1,1};
  	// std::vector<double> a (aArray, lArray + sizeof(lArray) / sizeof(double) );
		}
	}
}


void ROF3D::proxTVvOnTau(const cv::Mat &input,cv::Mat &output)

// // this function  computes prox(1/m_tau*TVv(input): it resolves the problem argmin( Sigma |v(i+1,j,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
// // the m_f is intialized before and refer to the object introduced to transform the problem of minimization to a ROF problem
	
	
	int sizeInputy=input.size[0];
	int sizeInputx=input.size[1];
	int sizeInputt=input.size[2];
	cv::Mat inputppk;
	std::vector<double> inputpjkVect;
	cv::Mat inputpjk;

	cv::Mat outputppk;
	std::deque<double> outputpjkDeque;

	for (int k=0;k<sizeInputt;k++)
	{
		inputppk=getLayer3D(input,k);
		outputppk=getLayer3D(output,k);

		for (int j=0;j<sizeInputx;j++)
		{
			inputpjk=getLayer2D(inputppk,j);inputpjk.copyTo(inputpjkVect);//matrix.col(0).copyTo(vec);
			// outputpjk=getLayer2D(outputppk,i);

			// double * inputpjk=inputppk.ptr<double>(j);
			// inputpjkVect=std::vector<double>(inputpjk,inputpjk+sizeInputy);

			// double * outputpjk=inputppk.ptr<double>(j);

			// // double * gij=gi.ptr<double>(j);
			// // gijVect=std::vector<double>(gij,gij+sizeInputt-1);

			ROF rof=ROF(m_tau,inputpjkVect);// this function resolves argmin_{u}( Sigma |u(k+1)-u(k)|+m_tau/2*Sigma|u(k)-inputpjk(k)|^2)
			// //this function resolve argmin_{v(i,j,.)}( Sigma_{k} g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+m_tau/2*Sigma_{k} |v(i,j,k)-l(i,j,k)|^2) with v(i,j,k) on R
			outputpjkDeque=rof.getSolution(false);
			// (uchar *) adress=output.data+output.size[1]*j+output.size[2]*k;
			// int limit=output.step[0]* output.step[1]*j+output.step[2]*k
			for (int i=0;i<sizeInputy;i++)
			{
				output.at<double>(cv::Vec<int,3>(i,j,k))=outputpjkDeque[i];
			}
			// // printContentsOf3DCVMat(getRow2D(inputi,j),true,"fij.txt");
			// for (int i=0;k<sizeInputy;k++)
			// {
			// 	outputpjk[k]=outputijDeque[k];
			// }

			// ROF(m_m_tau,fijVect);
			// double bArray[] = {1,1,1,1,1};
  	// std::vector<double> a (aArray, lArray + sizeof(lArray) / sizeof(double) );
		}
	}
}



void ROF3D::proxTVhOnTau(const cv::Mat &input,cv::Mat &output)

// // this function  computes prox(1/m_tau*TVv(input): it resolves the problem argmin( Sigma |v(i,j+1,k)-v(i,j,k)|+m_tau/2*Sigma|v(i,j,k)-input(i,j,k)|) with v(i,j,k) on R
{
// // the m_f is intialized before and refer to the object introduced to transform the problem of minimization to a ROF problem
	
	
	int sizeInputy=input.size[0];
	int sizeInputx=input.size[1];
	int sizeInputt=input.size[2];
	cv::Mat inputi;
	std::vector<double> inputipkVect;
	cv::Mat inputipk;
	// cv::Mat gi;
	// std::vector<double> gipkVect;
	// std::vector<double> inputipkVect;
	// int size[3] = { sizeInputy,sizeInputX,sizeInputt};
	// output=cv::Mat(3,size,CV_64FC1, 0.0);
	// cv::Mat output;
	cv::Mat outputi;
	std::deque<double> outputipkDeque;

	for (int i=0;i<sizeInputy;i++)
	{
		inputi=getRow3D(input,i);
		outputi=getRow3D(output,i);

		for (int k=0;k<sizeInputt;k++)
		{
			// double * inputipk=inputi.ptr<double>(j);
			// inputipkVect=std::vector<double>(inputij,inputij+sizeInputt);
			inputipk=getLayer2D(inputi,k);inputipk.copyTo(inputipkVect);
			// double * outputij=outputi.ptr<double>(j);

			// double * gij=gi.ptr<double>(j);
			// gipkVect=std::vector<double>(gij,gij+sizeInputt-1);

			ROF rof=ROF(m_tau,inputipkVect);// this function resolves argmin_{u}( Sigma gij(k)|u(k+1)-u(k)|+m_tau/2*Sigma|u(k)-inputij(k)|^2)
			//this function resolve argmin_{v(i,j,.)}( Sigma_{k} g(i,j,k)*|v(i,j,k+1)-v(i,j,k)|+m_tau/2*Sigma_{k} |v(i,j,k)-l(i,j,k)|^2) with v(i,j,k) on R
			outputipkDeque=rof.getSolution(false);
			// printContentsOf3DCVMat(getRow2D(inputi,j),true,"fij.txt");
			// for (int k=0;k<sizeInputt;k++)
			// {
			// 	outputij[k]=outputipkDeque[k];
			// }
			for (int j=0;j<sizeInputx;j++)
			{
				output.at<double>(cv::Vec<int,3>(i,j,k))=outputipkDeque[j];
			}
		}
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
