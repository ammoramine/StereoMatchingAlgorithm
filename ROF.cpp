#include "ROF.h"

ROF::ROF(const double &tau,const std::vector<double> &l,const std::vector<double> &costij)
// this function resolve argmin( Sigma cost(ij)|v(i+1)-v(i)|+tau/2*Sigma|vi-li|^2) with vi on R
// costij should be smaller than l on size by one element, and should contains postive elements
{
	m_tau=tau;
	int lengthROF=l.size();
	m_l=std::vector<double>(l);
	m_a=std::vector<double>(lengthROF,tau);
	m_b=std::vector<double>(lengthROF,tau);for (int i=0;i<l.size();i++) m_b[i]*=l[i];
	
	m_index_root=m_a.size()-1;
	m_omegaijPlus=std::deque<double>(costij.begin(),costij.end());m_omegaijPlus.push_back(0);
	//the last element should be zero w the other elements shold be equal to costij

	m_omegaijMinus.resize(lengthROF);for (int i=0;i<lengthROF;i++) m_omegaijMinus[i]=-m_omegaijPlus[i];
	//the last element should be zero w the other elements shold be equal to the costij
	
	computeROF();

}
ROF::ROF(const double &tau,const std::vector<double> &l)
// this function resolve argmin( Sigma |v(i+1)-v(i)|+tau/2*Sigma|vi-li|^2) with vi on R
{
	m_tau=tau;
	int lengthROF=l.size();
	m_l=std::vector<double>(l);
	m_a=std::vector<double>(lengthROF,tau);
	m_b=std::vector<double>(lengthROF,tau);for (int i=0;i<l.size();i++) m_b[i]*=l[i];
	
	m_index_root=m_a.size()-1;
	m_omegaijMinus=std::deque<double>(lengthROF-1,-1);m_omegaijMinus.push_back(0);//the last element should be zero w the other elements shold be equal to 1
	m_omegaijPlus=std::deque<double>(lengthROF-1,1);m_omegaijPlus.push_back(0);//the last element should be zero w the other elements shold be equal to 1
	computeROF();

}

double ROF::computeCostForArgument1DROF(const std::deque<double> &argument)
 //argument should de declared and initialized, this function compute the value of the minimized function of the proximal operator of 1/tau*TVL for the argument argument and the cost m_g:
 // it computes Sigma |v(i+1)-v(i)|+tau/2*Sigma|v(i)-l(i)|  with v the argument and the cost
 {
 	double result=0;
 	// std::vector<double> l(m_b);for (int i=0;i<l.size();i++) l[i]/=m_a[i];
	for (int i=0;i<argument.size();i++)
	{
		result+=0.5*m_tau*pow(argument[i]-m_l[i],2);
	}
	for (int i=0;i<argument.size()-1;i++)
	{
		result+=std::abs(m_omegaijPlus[i]*(argument[i+1]-argument[i]));
	}
	return result;
 }

 void ROF::testMinimialityOfSolution(int numberOfTests,double margin)
{
	double costArgmin=computeCostForArgument1DROF(m_x);
	std::deque<double> argument;
	bool succes=true;
	std::cout<<"\n computing argmin of Sigma |v(i+1)-v(i)|+m_tau/2*Sigma|v(i)-m_l(i)|^2 with m_l equal to : "<<std::endl;
		std::cout<<"  ( ";for (int i=0;i<m_l.size();i++) std::cout<<m_l[i]<<",";std::cout<<" ) \n";
		std::cout<<" and m_tau (that should be strictly positive ) equal to "<<m_tau<<std::endl;

	std::cout<<"\n the problem is equivalent to computing argmin of Sigma |v(i+1)-v(i)|+1/2*Sigma a(i)*(x(i)-b(i)/a(i))^2 \n with a equal to : "<<std::endl;

	std::cout<<"  ( ";
	for (int i=0;i<m_a.size();i++) 
		{
			std::cout<<m_a[i]<<",";
		}
	std::cout<<" ) \n";
	for (int i=0;i<m_a.size();i++) 
		{
		if (m_a[i]<=0)
			{
				std::cout<<"all the element a(i) should be strictly positive to minimi the problem \n";return;
			}
		}
	std::cout<<"with b equal to"<<std::endl;
	std::cout<<"  ( ";for (int i=0;i<m_b.size();i++) std::cout<<m_b[i]<<",";std::cout<<" ) \n";

		std::cout<<" and the solution is  equal to "<<std::endl;
		std::cout<<"  ( ";for (int i=0;i<m_x.size();i++) std::cout<<m_x[i]<<",";std::cout<<" ) \n";
		std::cout<<" with a cost of :"<<costArgmin<<std::endl;

	srand (time(NULL));
	for (int i=0;i<numberOfTests;i++)
	{
		argument=std::deque<double>(m_x);
		for (int j=0;j<argument.size();j++) argument[j]+=(double(rand())/double(RAND_MAX)-0.5)*margin/0.5;

		double costArgument=computeCostForArgument1DROF(argument);
		
		std::cout<<"\n cost of the minimum : "<<costArgmin<<" cost of a neighbor argument : "<<costArgument<<" and difference :"<<costArgument-costArgmin <<"  should be or not : "<<(costArgument-costArgmin>=0)<<std::endl;
		succes=(costArgument-costArgmin>=0);
		if(succes==false)
			{
				std::cout<< "problem with the last argument"<<std::endl;break;
			}
		std::cout<<" the argument is ( ";
		for (int i=0;i<argument.size();i++)
		{
			std::cout<<argument[i]<<",";
		}
		std::cout<<" ) \n";
	}
	if (succes==true)
	{
		std::cout<<"\n \n for the "<<numberOfTests<<" generated arguments the minimizer succeded to minimize the cost function"<<std::endl;
	}
}


ROF::ROF(const std::vector<double> &a,const std::vector<double> &b)
// a should be a vector of strictly positive elements
{
	// double tau=0.5;
	// int lArray[] = {1,1,1,1,1};
  	// std::vector<double> l (lArray, lArray + sizeof(lArray) / sizeof(int) );
	// m_a.resize(l.size());m_b.resize(l.size());
	m_a=std::vector<double>(a);//for(int i=0;i<l.size();i++) m_a[i]*=tau
	m_b=std::vector<double>(b);//for(int i=0;i<l.size();i++) m_b[i]*=tau;
	
	m_index_root=m_a.size();
	m_omegaijMinus=std::deque<double>(m_index_root,-1);m_omegaijMinus.push_back(0);//the last element should be zero w the other elements shold be equal to 1
	m_omegaijPlus=std::deque<double>(m_index_root,1);m_omegaijPlus.push_back(0);//the last element should be zero w the other elements shold be equal to 1
	computeROF();
	
	// computeROFForwardPass(m_a,m_b,m_lambdaMinVect,m_lambdaPlusVect);
	// computeROFBackwardPass(m_lambdaMinVect,m_lambdaPlusVect,x);
	// for (int i=0;i<x.size();i++) std::cout<<x[i]<<std::endl;
}

void ROF::initForwardPass()
{
	m_index_current_node=0;
	m_aBar_current_node=m_a[0];
	m_message.push_back(0);
	m_lambdaMinusCurrentNode=(m_b[0]+m_omegaijMinus[0])/(m_a[0]+m_message[0]);
	m_lambdaPlusCurrentNode=(m_b[0]+m_omegaijPlus[0])/(m_a[0]+m_message[0]);
	m_lambdaMinVect.push_back(m_lambdaMinusCurrentNode);
	m_lambdaPlusVect.push_back(m_lambdaPlusCurrentNode);
	m_message.push_front(m_lambdaMinVect[m_lambdaMinVect.size()-1]);m_message.push_front(-m_aBar_current_node);
	m_message.push_back(m_lambdaPlusVect[m_lambdaPlusVect.size()-1]);m_message.push_back(-m_aBar_current_node);
}
void ROF::computeSmallestIndexAndLambdaMinus()
// this function compute the index of the smallest breakpoints :element lambda for which the value of the message if above wikMin.
// If all the elements are under wikMin, the index is set to m_message.size[].the computation begin from the left. 
// Given the form of the message <s0,lambda1,s1,....,lambdai,si,....lambdat,st>, this index should always be odd, it's valeu is between 1 and m.size()

{
	double wkiMin=m_omegaijMinus[m_index_current_node-1];
	double wikMin=m_omegaijMinus[m_index_current_node];
	double ai=m_a[m_index_current_node];
	double bi=m_b[m_index_current_node];

	double lambdaCurrent;
	double lambdaCurrentPlus1;
	double sCurrent;//double sCurrentm1;double sCurrentp1;

	lambdaCurrent=m_message[1];
	double mLambdaCurrent=wkiMin+ai*lambdaCurrent-bi;

	int p=1;// thr index on m of th current lambda

	while(mLambdaCurrent<wikMin and p<m_message.size()-2)//the lowest possible index for lambda is 1 and the bigger one is  m.size()-2, if it reachs m_message.size(), it means all the elements are below wikMin
	{
		lambdaCurrent=m_message[p];
		lambdaCurrentPlus1=m_message[p+2];
		sCurrent=m_message[p+1];
		mLambdaCurrent+=(sCurrent+m_aBar_current_node)*(lambdaCurrentPlus1-lambdaCurrent);
		p+=2;// if mLambdaCurrent<wikMin the index is upper otherwise p is the value of the true index
	}
	// two case are possible mLambdaCurrent is still below wikMin, it means for all the breakpoints, we have mLambdaCurrent<wikMin
	if(mLambdaCurrent<wikMin) // m_message(lambda) is below wikMin for all the breakpoints
	{
		p+=2;// in order to keep from the message: message <s0,lambda1,s1,....,lambdai,si,....lambdat,st> only <st> in the function updateMessage()
	}
	// otherwise m_message(lambda) is above for lambda equal to the last breakpoint then we keep p to its value


	m_smallestIndexCurrentNode=p;
	// lastComputedMLambda=mLambdaCurrent;
	m_lambdaMinusCurrentNode=computeLambdaMinus(mLambdaCurrent);
}
double ROF::computeLambdaMinus(const double &lastComputedMLambda)
{
	// lastComputedMLambda is the last computed m(lambda) , lambda being the breakpoints for which m(lambda) is upper than wijMinus or the biggest breakpoint of the message
	double wijMinus=m_omegaijMinus[m_index_current_node];
	double lambdaMinusCurrentNode;

	if (m_smallestIndexCurrentNode==m_message.size())
	{
		

		double mLambdamax=lastComputedMLambda;// the last
		double lambdaMax=m_message[m_smallestIndexCurrentNode-2];

		double slopeAfterLambdaMax=m_message[m_smallestIndexCurrentNode-1]+m_aBar_current_node;

		lambdaMinusCurrentNode=lambdaMax+(wijMinus-mLambdamax)/(slopeAfterLambdaMax);// lambdaMin is after the biggest breakpoint
	}	
	else // m_smallestIndexCurrentNode is between 1 and m.size()-2
		// the last computed m_message(lambda) is above wijMinus so lambdaMin is juste below the breakpoint
	{
		double lambdap=m_message[m_smallestIndexCurrentNode];
		double mLambdap=lastComputedMLambda;

		double slopeBeforeLambda=m_aBar_current_node+m_message[m_smallestIndexCurrentNode-1];

		lambdaMinusCurrentNode=lambdap+(wijMinus-mLambdap)/slopeBeforeLambda;
	}
	return lambdaMinusCurrentNode;
}


void ROF::computeBiggestIndexAndLambdaPlus()
// this function compute the index of the biggest breakpoint :element lambda for which the value of the message if under wikPlus.
// If all the elements are above wikPlus, the index is set to -1.the computation begin from the right. 
// Given the form of the message <s0,lambda1,s1,....,lambdai,si,....lambdat,st>, this index should always be odd, it's valeu is between -1 and m.size()-2

{
	double wkiPlus=m_omegaijPlus[m_index_current_node-1];
	double wikPlus=m_omegaijPlus[m_index_current_node];
	double ai=m_a[m_index_current_node];
	double bi=m_b[m_index_current_node];

	double lambdaCurrent;
	double lambdaCurrentPlus1;
	double sCurrent;//double sCurrentm1;double sCurrentp1;

	lambdaCurrent=m_message[m_message.size()-2];
	double mLambdaCurrent=wkiPlus+ai*lambdaCurrent-bi;

	int r=m_message.size()-2;// thr index on m of th current lambda

	while(mLambdaCurrent>wikPlus and r>1)//the lowest possible index for lambda is 1 and the bigger one is  m_message.size()-2, if it reachs -1, it means all the elements are above wikPlus
	{
		r-=2;// if mLambdaCurrent>wikPlus the index is lower otherwise r is the value of the true index
		lambdaCurrent=m_message[r];
		lambdaCurrentPlus1=m_message[r+2];
		sCurrent=m_message[r+1];
		mLambdaCurrent-=(sCurrent+m_aBar_current_node)*(lambdaCurrentPlus1-lambdaCurrent);
	}
	// two case are possible mLambdaCurrent is still above wikPlus, it means for all the breakpoints, we have mLambdaCurrent>wikPlus
	if(mLambdaCurrent>wikPlus) // m_message(lambda) is above wikPlus for all the breakpoints
	{
		r-=2;// in order to keep from the message: message <s0,lambda1,s1,....,lambdai,si,....lambdat,st> only <s0> in the function updateMessage()
	}
	// otherwise m_message(lambda) is under for lambda equal to the smallest breakpoint then we keep r to its value


		m_biggestIndexCurrentNode=r;
		// double lastComputedMLambda=mLambdaCurrent;
		m_lambdaPlusCurrentNode=computeLambdaPlus(mLambdaCurrent);
}

double ROF::computeLambdaPlus(const double &lastComputedMLambda)
{
	// lastComputedMLambda is  the last computed m(lambda) , lambda being the breakpoints for which m(lambda) is below than wijPlus or the smallest breakpoint of the message
	double wijPlus=m_omegaijPlus[m_index_current_node];
	double lambdaPlusCurrentNode;
	if (m_biggestIndexCurrentNode==-1)// all the element are bigger then wijPlus
	{
		

		double mLambdaMin=lastComputedMLambda;// the value of the message for the smallest value of lambda
		double lambdaMin=m_message[m_biggestIndexCurrentNode+2];

		double slopeBeforeLambdaMin=m_message[m_biggestIndexCurrentNode+1]+m_aBar_current_node;

		lambdaPlusCurrentNode=lambdaMin+(wijPlus-mLambdaMin)/(slopeBeforeLambdaMin);// lambdaMax is after the biggest breakpoint
	}	
	else // m_smallestIndexCurrentNode is between 1 and m.size()-2
		// the last computed m_message(lambda) is above wijMinus so lambdaMax is juste below the breakpoint
	{
		double lambdar=m_message[m_biggestIndexCurrentNode];
		double mLambdar=lastComputedMLambda;

		double slopeAfterLambda=m_aBar_current_node+m_message[m_biggestIndexCurrentNode+1];

		lambdaPlusCurrentNode=lambdar+(wijPlus-mLambdar)/slopeAfterLambda;
	}
	return lambdaPlusCurrentNode;
}



void ROF::computeROF()
{
	
	initForwardPass();

	// printInformationsForNode();
	for (m_index_current_node=1;m_index_current_node<=m_index_root;m_index_current_node++)
	{
		iterateForwardPass();
		// printInformationsForNode();

	}
	computeROFBackwardPass();
}

void ROF::iterateForwardPass()
{
	m_aBar_current_node+=m_a[m_index_current_node];
	computeSmallestIndexAndLambdaMinus();
	computeBiggestIndexAndLambdaPlus();
	m_lambdaMinVect.push_back(m_lambdaMinusCurrentNode);
	m_lambdaPlusVect.push_back(m_lambdaPlusCurrentNode);
	updateMessage();
}




void ROF::computeROFBackwardPass()
//const std::vector<double> &lambdaMinVect,const std::vector<double> &lambdaPlusVect,std::deque<double> &x)
{
	if ( std::abs(m_lambdaMinVect[m_lambdaMinVect.size()-1]-m_lambdaPlusVect[m_lambdaPlusVect.size()-1])>0.0001 )
			{

		throw std::invalid_argument( "problem with the forward Pass: lambdaMin(n,n+1) and lambdaMax(n,n+1) should be equal" );
	}
	m_x.push_front(m_lambdaMinVect[m_lambdaMinVect.size()-1]);
	double lambdaMin;double lambdaPlus;
	for (int i=m_lambdaMinVect.size()-2;i>=0;i--)
		{
			lambdaMin=m_lambdaMinVect[i];lambdaPlus=m_lambdaPlusVect[i];
			double h=clip(m_x[0],lambdaMin,lambdaPlus);
			m_x.push_front(h);
		}
}

double ROF::clip(const double &x,const double &lambdaMinus,const double &lambdaPlus)
{
	return std::min(std::max(x,lambdaMinus),lambdaPlus);
}



void ROF::printInformationsForNode()
{
	std::cout<<"\n message mi and mi(i+1) for node i = : "<<m_index_current_node<<"\n";
	std::cout<<"( ";
	for (int i=0;i<m_message.size();i++)
	{
		std::cout<<m_message[i]<<",";
	}
	std::cout<<" ) \n";
	std::cout<<" value of lambdaPlus : "<<m_lambdaPlusCurrentNode<<" value of lambdaMinus : "<<m_lambdaMinusCurrentNode<<std::endl;
}

void ROF::updateMessage()
{
	// m_smallestIndex is between 1 and m.size() and is odd 
	// biggestIndex is between -1 and m.size()-2 and is odd 
	
	//
	std::deque<double>::iterator newlowestIndex=m_message.begin()+m_smallestIndexCurrentNode-1;
	std::deque<double>::iterator newBiggestIndex=m_message.begin()+m_biggestIndexCurrentNode+1;
	// std::deque<double> mNew(newlowestIndex,newBiggestIndex+1);
	// m=mNew;
	m_message=std::deque<double>(newlowestIndex,newBiggestIndex+1);

	m_message.push_front(m_lambdaMinusCurrentNode);m_message.push_front(-m_aBar_current_node);
	m_message.push_back(m_lambdaPlusCurrentNode);m_message.push_back(-m_aBar_current_node);
}


std::deque<double> ROF::getSolution(bool printIt)
{
	if(printIt==true)
	{
	std::cout<<"\n minimization of f with f(x)=Sigma_{i}(fi(xi))+Sigma_{i,i+1}(f_{i,i+1}(x_{i+1}-x{i})) \n and fi of the form 1/2*a_{i}x^2-b{i}x and f_{i,i+1}(x) of the form |x|)";
	
	std::cout<<"\n here we have the following values for a_{i}";
	std::cout<<"( ";
	for (int i=0;i<m_a.size();i++)
	{
		std::cout<<m_a[i]<<",";
	}
	std::cout<<" ) \n";
	
	std::cout<<"\n and we have the following values for b_{i}";
	std::cout<<"( ";
	for (int i=0;i<m_b.size();i++)
	{
		std::cout<<m_b[i]<<",";
	}
	std::cout<<" ) \n";

	std::cout<<"\n the solution of the onedimensionnal ROF problem "<<"\n";
	
		std::cout<<"( ";
		for (int i=0;i<m_x.size();i++)
		{
			std::cout<<m_x[i]<<",";
		}
		std::cout<<" ) \n";
	}
	return m_x;
}
// double clip(const double &x,const double &lambdaMinus,const double &lambdaPlus)
// {
// 	return std::min(std::max(x,lambdaMinus),lambdaPlus);
// }
