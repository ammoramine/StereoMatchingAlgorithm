#ifndef ROFBIS_INCLUDED_H
#define ROFBIS_INCLUDED_H
#include <vector>
#include <deque>
#include <stdexcept>
#include <stdlib.h>
#include <iostream>


class ROF 
{
	public:
		ROF(const double &tau,const double &l,int lengthROF);
		ROF(const std::vector<double> &a,const std::vector<double> &b);
		void computeROF();
		void initForwardPass();
		void iterateForwardPass();
		void computeROFBackwardPass();

		void computeSmallestIndexAndLambdaMinus();
		void computeBiggestIndexAndLambdaPlus();
		double computeLambdaMinus(const double &lastComputedMLambda);// should be called only by computeSmallestIndexAndLambdaMinus
		double computeLambdaPlus(const double &lastComputedMLambda);// should be called only by computeBiggestIndexAndLambdaPlus
		
		void updateMessage();
		void printInformationsForNode();
        static double clip(const double &x,const double &lambdaMinus,const double &lambdaPlus);
        std::deque<double> getSolution(bool printIt=false);

	private:
		std::vector<double> m_a;
		std::vector<double> m_b;
		std::deque<double> m_x;// the result
		std::deque<double> m_message;
		// the vector m_message is a reprentation for the message mi in the form <s0,lambda1,s1,....,lambdai,si,....lambdat,st>, m_message represents a  piecewise affine function
		// sp are the breakpoints  , and lambdap the consecutives slopes
		//the size of m_message should always be odd

		std::deque<signed int> m_omegaijMinus;
		std::deque<signed int> m_omegaijPlus;
		std::vector<double> m_lambdaMinVect;
		std::vector<double> m_lambdaPlusVect;
		int m_index_root;
		int m_index_current_node;// the index of the current node taking value from 0 to m_index_root;
		double m_aBar_current_node;// it's the valeu of aBar for mi and mij j=i+1 and i being the m_index_current_node
		
		signed int m_smallestIndexCurrentNode;signed int m_biggestIndexCurrentNode;// respectively the index of the smallest breakpoints above wijMin and the biggest breakpoints under wijPlus, with wijMin=omegaijMinus[m_index_current_node] wijPlus=omegaijPlus[m_index_current_node], at the nd of the iteration

		double m_lambdaMinusCurrentNode;double m_lambdaPlusCurrentNode; // defined by message(m_lambdaMin_current_node)=wijMin and message(m_lambdaPlus_current_node)=wijPlus

		// double m_lastComputedMLambda;// the last computed value of m(lambda) during the iterations
};

#endif