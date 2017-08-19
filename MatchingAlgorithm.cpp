#include "MatchingAlgorithm.h"

MatchingAlgorithm::MatchingAlgorithm(const cv::Mat &image1,const cv::Mat &image2,std::string dataTermOption,int t_size,signed int offset,int Niter,std::string path_to_disparity)
// tahe as input two gray images
{
	m_image1=new cv::Mat(image1.size(),image1.type());
	m_image2=new cv::Mat(image2.size(),image2.type());
	image1.copyTo(*m_image1);
	image2.copyTo(*m_image2);
	m_y_size=m_image1->size().height;
	m_x_size=m_image1->size().width;
	m_path_to_disparity=path_to_disparity;
	// m_disparity=cv::Mat(m_y_size,m_x_size,CV_64FC1,0.0);
	
	m_mu = 75.0/255.0;
	m_tau = 1.0/sqrt(12.0) ;
	m_sigma = 1.0/sqrt(12.0) ;
	m_s=0.5;


	m_Niter = Niter;//700;
	m_factor=0.05;
	m_t_size = t_size ; //need to impose that m_t_size is smaller than m_x_size
	m_offset=offset;
	// m_offset=int(-double(m_t_size)/2.0);
	// m_disorderedImages= new std::vector<cv::Mat>;
	// m_g.resize(m_t_size);

	m_iteration=0; // at the begining no iteration is done
	m_dataTermOption=dataTermOption;
	printProperties();
	data_term_effic();
	init();

	launch();


	disparity_estimation();

}

MatchingAlgorithm::~MatchingAlgorithm()
{
	delete(m_image1);
	delete(m_image2);
}


void MatchingAlgorithm::data_term() //(Im1,Im2,Nt,mu)
{

// calcul le cost volume mais dans ce cas le déplacement n'est permis que dans un seul sens, il faut corriger ca !

// calcule le cost-volume g(i,j,k) = mu * sum_{R,G,B} abs(Im1(i,j) - Im2(i,j-k))
// %Im1 : image de gauche
// %Im2 : image de droite (deplacement de camera de gauche vers la droite)


	int size[3] = { m_y_size, m_x_size, m_t_size };
	m_g=cv::Mat(3, size, CV_64FC1, 500.0);

		for (int i=0;i<m_y_size;i++)
		{
			// const uchar* data_in_1_line_i= m_image1->ptr<uchar>(i);
			// const uchar* data_in_2_line_i= m_image2->ptr<uchar>(i);
			// uchar* data_out_line_i= m_g[k].ptr<uchar>(i);
			// std::cout<<"data 1 : "<<(*data_in_1_line_i) <<std::endl;
			// std::cout<<"data 2 : "<<(*data_in_2_line_i) <<std::endl;
			for (int k=0;k<m_t_size;k++)
				{
					for (int j=k;j<m_x_size;j++)
						{

					// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							m_g.at<double>(i,j,k)=m_mu*abs(m_image1->at<double>(i,j)-m_image2->at<double>(i,j-k));
						}
				}
		}
	
}


void MatchingAlgorithm::data_term_effic() //(Im1,Im2,Nt,mu)
{

// calcul le cost volume mais dans ce cas le déplacement n'est permis que dans un seul sens, il faut corriger ca !

// calcule le cost-volume g(i,j,k) = mu * sum_{R,G,B} abs(Im1(i,j) - Im2(i,j-k))
// %Im1 : image de gauche
// %Im2 : image de droite (deplacement de camera de gauche vers la droite)

//m_t_size should be smaller than m_x_size and m_y_size

int size[3] = { m_y_size, m_x_size, m_t_size };
m_g=cv::Mat(3, size, CV_64FC1, 500.0);
// printContentsOf3DCVMat(*m_image1,false);
// cv::waitKey(100);
// printContentsOf3DCVMat(*m_image2,false);return;
if (m_dataTermOption=="absdiff")
	{

		for (int i=0;i<m_y_size;i++)
		{
			// double * deltaxPtr=deltax.ptr<double>(0);
			double * m_image1iPtr= m_image1->ptr<double>(i);
			double * m_image2iPtr= m_image2->ptr<double>(i);
			cv::Mat m_gi=getRow3D(m_g,i);
			for (int j=0;j<m_x_size;j++)
			{
				double * m_gij=m_gi.ptr<double>(j);
				int maxk=std::min(j-m_offset,m_t_size-1);
				int mink=std::max(j-m_offset-m_x_size,0);
				for(int k=mink;k<=maxk;k++)
				{
			// for (int km
					m_gij[k]=m_mu*abs(m_image1iPtr[j]-m_image2iPtr[j-k-m_offset]);
				}
				// delete m_gij;
			}
		}
			// delete m_image1iPtr;
			// delete m_image2iPtr;
	}
else if (m_dataTermOption=="census")
	{
		data_term_census(*m_image1,*m_image2,m_g);
		m_g=m_mu*m_g;
	}
else 
{
	throw std::invalid_argument( "intern problem on MatchingAlgorithm.cpp" );
}
}
cv::Mat  MatchingAlgorithm::projCh(const cv::Mat &v)
{

 
// %projection sur [0,1], avec v(i,j,1) = 1 et v(i,j,end) = 0
// this function project a vector of elements of size  m_y_size,m_x_size,m_t_size on the convex C, the entry should be of the form of std::vector<cv::Mat> of size  m_y_size,m_x_size,m_t_size  and also the ouput, the type must be of  CV_64FC1
	// cv::Mat vproj;
	int size[3] = { v.size[0],v.size[1],v.size[2]};
	cv::Mat vproj(3, size, CV_64FC1, 0.0);
	cv::Mat dst;
	cv::max(v,0.0, dst);
	cv::min(dst,1.0,vproj);
	for (int i = 0; i < vproj.size[0]; i++)
	{	
  		for (int j = 0; j < vproj.size[1]; j++)
    	{	
			vproj.at<double>(i,j,0)=1.0;//cv::Mat::ones(m_y_size,m_x_size,CV_64FC1);
			vproj.at<double>(i,j,vproj.size[2]-1)=0.0;//cv::Mat::zeros(m_y_size,m_x_size,CV_64FC1);
		}
	}
	return vproj;
}

cv::Mat  MatchingAlgorithm::projCh_effic(const cv::Mat &v)//,std::vector<cv::Mat> &vproj) //= projC(v)
{

 
// %projection sur [0,1], avec v(i,j,1) = 1 et v(i,j,end) = 0
// this function project a vector of elements of size  m_y_size,m_x_size,m_t_size on the convex C, the entry should be of the form of std::vector<cv::Mat> of size  m_y_size,m_x_size,m_t_size  and also the ouput, the type must be of  CV_64FC1
	// cv::Mat vproj;
	int size[3] = { v.size[0],v.size[1],v.size[2]};
	cv::Mat vproj(3, size, CV_64FC1, 0.0);
	// for (int i = 0; i < 100; i++)
  		// for (int j = 0; j < 100; j++)
    		// for (int k = 0; k < 3; k++) 
	// 			std::cout<<"i,j,k : "<<i<<j<<k<<" "<<M.at<double>(i,j,k)<<std::endl;

	// vproj.resize(v.size());
	
	// for (int k=0;k<m_t_size;k++)
	// {
	cv::Mat dst;
	cv::max(v,0.0, dst);
	cv::min(dst,1.0,vproj);
		// min()
	// }
	cv::Mat vproji;
	// cv::Mat vprojij;
	// double * vprojij;
	for (int i = 0; i < vproj.size[0]; i++)
	{	
		vproji=getRow3D(vproj,i);
  		for (int j = 0; j < vproj.size[1]; j++)
    	{	
    		// double * vprojij=getRow2D(vproji,j);
    		double * vprojij=vproji.ptr<double>(j);
    		vprojij[0]=1.0;
    		vprojij[vproj.size[2]-1]=0.0;
    		// for (int k = 0; k < 3; k++) 
			// vproj.at<double>(i,j,0)=1.0;//cv::Mat::ones(m_y_size,m_x_size,CV_64FC1);
			// vproj.at<double>(i,j,vproj.size[2]-1)=0.0;//cv::Mat::zeros(m_y_size,m_x_size,CV_64FC1);
		}
	}
	return vproj;
}


cv::Mat  MatchingAlgorithm::projKh(const cv::Mat &phi,const cv::Mat &g)
// phi if of size {3, m_y_size, m_x_size, m_t_size }, the first term id phiy, the second phix and the last one phit
{
	// int size1[4] = {3, m_y_size, m_x_size, m_t_size };
	// int size2[3] = {m_y_size, m_x_size, m_t_size };
	int size1[4] = {phi.size[0], phi.size[1], phi.size[2], phi.size[3] };
	// int size2[3] = {phi.size[1], phi.size[2], phi.size[3] };
// apply q_x = phi_x ; q_y = phi_y ; q_t = phi_t + g ;
	cv::Mat q;//(4, size1, CV_64FC1, 0.0);
	phi.copyTo(q);
	cv::Mat qy=getRow(q,0);
	cv::Mat qx=getRow(q,1);
	cv::Mat qt=getRow(q,2);
	qt=qt+g;


// computing normqx = max(sqrt(q_x.^2+q_y.^2),0.00001) ; the 0.00001 is to avoid the division problem
	cv::Mat normqyx=qx.mul(qx)+qy.mul(qy);
	cv::sqrt(normqyx,normqyx);
	cv::max(normqyx,0.00001,normqyx);

//declaration of qproj
	cv::Mat qproj(4, size1, CV_64FC1, 0.0);
	cv::Mat qprojy=getRow(qproj,0);
	cv::Mat qprojx=getRow(qproj,1);
	cv::Mat qprojt=getRow(qproj,2);
	
//computing qproj_x = q_x./(max(normqx,1)) , qproj_y = q_y./(max(normqx,1)) 
	cv::max(normqyx,1.0,qprojy);
	cv::max(normqyx,1.0,qprojx);

	qprojy=qy/qprojy; //cv::divide(qy,qprojy,qprojy)
	qprojx=qx/qprojx;
// computing , qproj_t = max(q_t,0) ;
	cv::max(qt,0.0,qprojt);

	cv::Mat phiProj=qproj;
	cv::Mat phiProjt=getRow(phiProj,2);phiProjt-=g;
	return phiProj;

}

cv::Mat  MatchingAlgorithm::projKh_effic(const cv::Mat &phi,const cv::Mat &g)
// phi if of size {3, Ny,Nx,Nt}, the first term is phiy, the second phix and the last one phit
{
	int size1[4] = {phi.size[0], phi.size[1], phi.size[2], phi.size[3] };
	int size2[3] = {phi.size[1], phi.size[2], phi.size[3] };

// apply q_x = phi_x ; q_y = phi_y ; q_t = phi_t + g ;
	// cv::Mat q;//(4, size1, CV_64FC1, 0.0);
	// phi.copyTo(q);
	// cv::Mat qy=getRow(q,0);
	// cv::Mat qx=getRow(q,1);
	// cv::Mat qt=getRow(q,2);
	// qt=qt+g;


// // computing the projection of q onto Kh
//declaration of qproj
	// cv::Mat qproj(4, size1, CV_64FC1, 0.0);
	cv::Mat phiProj;phi.copyTo(phiProj);
	cv::Mat phiProjy=getRow(phiProj,0);
	cv::Mat phiProjx=getRow(phiProj,1);
	cv::Mat phiProjt=getRow(phiProj,2);
	
	cv::Mat phiy=getRow(phi,0);
	cv::Mat phix=getRow(phi,1);
	cv::Mat phit=getRow(phi,2);

	for (int i=0;i<size2[0];i++)
	{
		cv::Mat phiProjyi=getRow3D(phiProjy,i);
		cv::Mat phiProjxi=getRow3D(phiProjx,i);
		cv::Mat phiProjti=getRow3D(phiProjt,i);

		cv::Mat phiyi=getRow3D(phiy,i);
		cv::Mat phixi=getRow3D(phix,i);
		cv::Mat phiti=getRow3D(phit,i);

		cv::Mat gi=getRow3D(g,i);
		for (int j=0;j<size2[1];j++)
		{
			double * phiProjyij=phiProjyi.ptr<double>(j);
			double * phiProjxij=phiProjxi.ptr<double>(j);
			double * phiProjtij=phiProjti.ptr<double>(j);

			double * phiyij=phiyi.ptr<double>(j);
			double * phixij=phixi.ptr<double>(j);
			double * phitij=phiti.ptr<double>(j);

			double * gij=gi.ptr<double>(j);
			for (int k=0;k<size2[2];k++)
			{
				phiProjtij[k]+=gij[k];
				double normqyx=sqrt(phiProjyij[k]*phiProjyij[k]+phiProjxij[k]*phiProjxij[k]);
				phiProjyij[k]=phiProjyij[k]/std::max(1.0,normqyx);
				phiProjxij[k]=phiProjxij[k]/std::max(1.0,normqyx);
				phiProjtij[k]=std::max(0.0,phiProjtij[k])-gij[k];
			}
			// delete
		}
	}
// //computing the projection of phi 

	// cv::Mat phiProj=qproj;
	// cv::Mat phiProjt=getRow(phiProj,2);phiProjt-=g;

	// cv::Mat phiproj(4, size1, CV_64FC1, 0.0);
	// cv::Mat phiy=getRow3D(phi,0);
	// cv::Mat phix=getRow3D(phi,1);
	// cv::Mat phit=getRow3D(phi,2);


	return phiProj;

}



	cv::Mat MatchingAlgorithm::gradh(const cv::Mat &v)
	//operateur defini par "Variational Models with Convex Regularization"

	// -input: div is of dimension 3 

	// -ouput: of dimension 4 and of size {3,v.size[0],v.size[1],v.size[2]}

	{
		// function [ delta_y,delta_x,delta_t ] = gradh(v)
		int size[4]= {3,v.size[0],v.size[1],v.size[2]};
		cv::Mat delta(4,size,CV_64FC1,0.0);
		for (int i = 0; i < v.size[0]-1; i++)
		{
			for (int j = 0; j < v.size[1]-1; j++)
			{
				for (int k = 0; k < v.size[2]-1; k++)
				{
					// cv::Vec<int,3>(i,j,k);
					delta.at<double>(cv::Vec<int,4>(0,i,j,k))=v.at<double>(i+1,j,k)-v.at<double>(i,j,k); // the y component
					delta.at<double>(cv::Vec<int,4>(1,i,j,k))=v.at<double>(i,j+1,k)-v.at<double>(i,j,k); // the x component
					delta.at<double>(cv::Vec<int,4>(2,i,j,k))=v.at<double>(i,j,k+1)-v.at<double>(i,j,k); // the t component
				}
			}
		}
		return delta;
	}



	cv::Mat MatchingAlgorithm::gradh_effic(const cv::Mat &v)
	// more efficient version of grad
	
	// -input: v is of dimension 3 

	// -ouput: of dimension 4 and of size {3,v.size[0],v.size[1],v.size[2]}

	{
		// function [ delta_y,delta_x,delta_t ] = gradh(v)
		int size[4]= {3,v.size[0],v.size[1],v.size[2]};
		cv::Mat delta(4,size,CV_64FC1,0.0);

		cv::Mat deltay=MatchingAlgorithm::getRow(delta,0);
		cv::Mat deltax=MatchingAlgorithm::getRow(delta,1);
		cv::Mat deltat=MatchingAlgorithm::getRow(delta,2);

		for (int i = 0; i < v.size[0]-1; i++)
		{
			cv::Mat deltayi=MatchingAlgorithm::getRow3D(deltay,i);
			cv::Mat deltaxi=MatchingAlgorithm::getRow3D(deltax,i);
			cv::Mat deltati=MatchingAlgorithm::getRow3D(deltat,i);

			cv::Mat vi=MatchingAlgorithm::getRow3D(v,i);
			cv::Mat vip1=MatchingAlgorithm::getRow3D(v,i+1);
			for (int j = 0; j < v.size[1]-1; j++)
			{
				double * deltayij=deltayi.ptr<double>(j);
				double * deltaxij=deltaxi.ptr<double>(j);
				double * deltatij=deltati.ptr<double>(j);			
				// cv::Mat vij=MatchingAlgorithm::getRow(vi,j);
				// cv::Mat vijp1=MatchingAlgorithm::getRow(vi,j+1);
				// cv::Mat vip1j=MatchingAlgorithm::getRow(vip1,j);
				double * vijPtr=vi.ptr<double>(j);
				double * vijp1Ptr=vi.ptr<double>(j+1);
				double * vip1jPtr=vip1.ptr<double>(j);			
				for (int k = 0; k < v.size[2]-1; k++)
				{
					deltayij[k]+=vip1jPtr[k]-vijPtr[k];
					deltaxij[k]+=vijp1Ptr[k]-vijPtr[k];
					deltatij[k]+=vijPtr[k+1]-vijPtr[k];
					// cv::Vec<int,3>(i,j,k);
					// delta.at<double>(cv::Vec<int,4>(0,i,j,k))=v.at<double>(i+1,j,k)-v.at<double>(i,j,k); // the y component
					// delta.at<double>(cv::Vec<int,4>(1,i,j,k))=v.at<double>(i,j+1,k)-v.at<double>(i,j,k); // the x component
					// delta.at<double>(cv::Vec<int,4>(2,i,j,k))=v.at<double>(i,j,k+1)-v.at<double>(i,j,k); // the t component
				}
			}
		}
		return delta;
	}

cv::Mat MatchingAlgorithm::divh(const cv::Mat &v)
// // -Description: a discretisation of -div and not div, we denote by divh the adjoint of the gradient operator

// // -input: div is of dimension 4 and of size {3,v.size[1],v.size[2],v.size[4]}

// // -ouput: of dimension 3 and of size {v.size[1],v.size[2],v.size[3]}



// // -comment: the scanning is not optimized ...
{
	int size[3]= {v.size[1],v.size[2],v.size[3]};
	cv::Mat divv(3,size,CV_64FC1,0.0);
	// cv::Mat divv()=r;
		for (int i = 1; i < size[0]; i++)
		{
			for (int j = 1; j < size[1]; j++)
			{
				for (int k = 1; k < size[2]; k++)
				{
					divv.at<double>(i,j,k)=v.at<double>(cv::Vec<int,4>(0,i-1,j,k))+v.at<double>(cv::Vec<int,4>(1,i,j-1,k))+v.at<double>(cv::Vec<int,4>(2,i,j,k-1));
					divv.at<double>(i,j,k)+=-v.at<double>(cv::Vec<int,4>(0,i,j,k));
					divv.at<double>(i,j,k)+=-v.at<double>(cv::Vec<int,4>(1,i,j,k));
					divv.at<double>(i,j,k)+=-v.at<double>(cv::Vec<int,4>(2,i,j,k));
				}
			}
		}
	return divv;
}

cv::Mat MatchingAlgorithm::divh_effic(const cv::Mat &v)
// -Description: a discretisation of -div and not div, we denote by divh the adjoint of the gradient operator
// a more efficient version of div 

// -input: div is of dimension 4 and of size {3,v.size[1],v.size[2],v.size[4]}

// -ouput: of dimension 3 and of size {v.size[1],v.size[2],v.size[3]}



// -comment: the scanning is not optimized ...
{
	int size[3]= {v.size[1],v.size[2],v.size[3]};
	cv::Mat divv(3,size,CV_64FC1,0.0);
	// cv::Mat divv()=r;
	cv::Mat vy=MatchingAlgorithm::getRow(v,0);
	cv::Mat vx=MatchingAlgorithm::getRow(v,1);
	cv::Mat vt=MatchingAlgorithm::getRow(v,2);

		for (int i = 1; i < size[0]; i++)
		{
			cv::Mat vyi = MatchingAlgorithm::getRow3D(vy,i);
			cv::Mat vxi = MatchingAlgorithm::getRow3D(vx,i);
			cv::Mat vti = MatchingAlgorithm::getRow3D(vt,i);
			cv::Mat vyim1 = MatchingAlgorithm::getRow3D(vy,i-1);
			cv::Mat divvi = MatchingAlgorithm::getRow3D(divv,i);
			for (int j = 1; j < size[1]; j++)
			{
				double * vyijPtr=vyi.ptr<double>(j);
				double * vxijPtr=vxi.ptr<double>(j);
				double * vtijPtr=vti.ptr<double>(j);

				double * vyim1jPtr=vyim1.ptr<double>(j);
				double * vxijm1Ptr=vxi.ptr<double>(j-1);
				// double * vtijPtr=vti.ptr<double>(j);
				double * divvij=divvi.ptr<double>(j);
				// cv::Mat vyim1j = MatchingAlgorithm::getRow2D(vyim1,j);
				// cv::Mat vyim1j = MatchingAlgorithm::getRow2D(vyim1,j);
				// cv::Mat divvi = MatchingAlgorithm::getRow(v,i);
				for (int k = 1; k < size[2]; k++)
				{
					// int kptr=k*sizeof();
					divvij[k]=-vyijPtr[k]-vxijPtr[k]-vtijPtr[k]+vyim1jPtr[k]+vxijm1Ptr[k]+vtijPtr[k-1];
					
				}
			}
		}
	return divv;
}

void MatchingAlgorithm::init()
{
	int sizeKh[4]= {3,m_y_size,m_x_size,m_t_size};
	int sizeCh[3]= {m_y_size,m_x_size,m_t_size};
	m_phih=cv::Mat(4,sizeKh,CV_64FC1,1.0);
	m_vbar=cv::Mat(3,sizeCh,CV_64FC1,1.0);
	m_v=cv::Mat(3,sizeCh,CV_64FC1,1.0);
	cv::randu(m_phih,0, 1.0);
	cv::randu(m_v,0, 1.0);
	
	m_v=projCh_effic(m_v);
	m_v.copyTo(m_vbar);
	
	
	m_phih=projKh(m_phih,m_g);
	m_gapInit=computePrimalDualGap();
	m_gap=m_gapInit;
}
double MatchingAlgorithm::computePrimalDualGap()
{
	int sizeCh[3]= {m_y_size,m_x_size,m_t_size};
	// int sizeCh1[3]= {m_y_size,m_x_size,m_t_size};
	cv::Mat delta = gradh_effic(m_v) ;
	cv::Mat divv = divh_effic(m_phih) ;

    cv::Mat v0 = (divv < 0.0);

	cv::Mat doubleV0;
    v0.convertTo(doubleV0, CV_64FC1); // converting for boolean to double
    // printContentsOf3DCVMat(getLayer(doubleV0,0));
	doubleV0 = projCh_effic(doubleV0) ;

	cv::Mat gapPerPixel(3,sizeCh,CV_64FC1,0.0);
	// cv::Mat temp(3,sizeCh,CV_64FC1,1.0);temp=temp-getRow(delta,1);
	cv::Mat deltay=getRow(delta,0);
	cv::Mat deltax=getRow(delta,1);
	cv::Mat deltat=getRow(delta,2);

	// cv::accumulateSquare(getRow(delta,0),gapPerPixel);cv::accumulateSquare(getRow(delta,1),gapPerPixel);
	gapPerPixel+=deltay.mul(deltay)+deltax.mul(deltax);
	cv::sqrt(gapPerPixel,gapPerPixel);


	// cv::Mat minus_mg=-m_g;
	// cv::Mat gapPerPixel2(3,sizeCh,CV_64FC1,0.0);
	// cv::accumulateProduct(-getRow(delta,2),m_g,gapPerPixel);cv::accumulateProduct(-divv,doubleV0,gapPerPixel);
	gapPerPixel+=-deltat.mul(m_g)-divv.mul(doubleV0);
	// gapPerPixel=gapPerPixel- gapPerPixel2;
	m_gap=cv::sum(gapPerPixel)[0];
	// return m_gap;
	// cv::scaleAdd(m_g,-1.0,phiproj.row(2),phiproj.row(2));
	// gap = sqrt(delta_x.^2 + delta_y.^2) - delta_t.*g - div.*v0 ;
// gap = sum(gap(:)) ;
	return m_gap;

}

void MatchingAlgorithm::iterate_algorithm()
{
    
	//computing phi^{n+1}
    cv::Mat delta = gradh_effic(m_vbar) ;
    // cv::Mat a=m_phih+delta*m_sigma;
	m_phih = projKh(m_phih + m_sigma*delta,m_g);    
	

	// computinv v^{n+1}
	
	cv::Mat m_v_previous;m_v.copyTo(m_v_previous); // we'll need the previous value

	cv::Mat divv= divh_effic(m_phih);
	m_v=projCh_effic(m_v - m_tau*divv);


	//computing vbar^{n+1}

	m_vbar=2.0*m_v - m_v_previous;	

	computePrimalDualGap();
	m_iteration+=1;
	std::cout<<"iteration number : "<<m_iteration<<" performed "<<" and gap equal to "<<m_gap<<std::endl;
}

void MatchingAlgorithm::launch()
//after the methodes init has been launched
{
	while( m_iteration < m_Niter and ( m_gap >= m_factor*m_gapInit ) )
	{
		iterate_algorithm();
		if(m_iteration%10==0) disparity_estimation();
	}
}


void MatchingAlgorithm::disparity_estimation()
{
	m_disparity=cv::Mat(m_y_size,m_x_size,CV_64FC1,0.0);
	cv::Mat thresholded = (m_v > m_s);
	// printContentsOf3DCVMat(getLayer(m_v,0),true,"firstlayerv");
	// printContentsOf3DCVMat(getLayer(m_v,4),true,"lastlayerv");
	// printContentsOf3DCVMat(getLayer(m_v,1),true,"secondlayerv");
	// printContentsOf3DCVMat(getLayer(m_v,2),true,"thirdlayerv");
	// printContentsOf3DCVMat(getLayer(m_v,3),true,"forthlayerv");
	int z=thresholded.size[2];

	int size[3] = { m_y_size, m_x_size, m_t_size };
	cv::Mat doubleThresholded=cv::Mat(3, size, CV_64FC1, 0.0);
	thresholded.copyTo(doubleThresholded);
    thresholded.convertTo(doubleThresholded, CV_64FC1);
    // printContentsOf3DCVMat(doubleThresholded,true);
	double zoomFactor=1/(double(doubleThresholded.size[2]));
	// cv::Mat thresholdedDouble;
    // thresholded.convertTo(thresholdedDouble, CV_64FC1);
    // int size[4]= {3,v.size[0],v.size[1],v.size[2]};
		// cv::Mat delta(4,size,CV_64FC1,0.0);
	for (int i = 0; i < doubleThresholded.size[0]; i++)
	{
		// cv::Mat doubleThresholdedi = MatchingAlgorithm::getRow3D(doubleThresholded,i);
		for (int j = 0; j < doubleThresholded.size[1]; j++)
		{
			// cv::Mat doubleThresholdedij = MatchingAlgorithm::getRow2D(doubleThresholdedi,j);
			for (int k = 0; k < doubleThresholded.size[2]; k++)
			{

				double value=doubleThresholded.at<double>(i,j,k);
				// value=zoomFactor*value;
				m_disparity.at<double>(i,j)+=value;
			// 	// if (thresholded.at<uchar>(i,j,k)!=0)
			// 	// {
				// m_disparity.at<double>(i,j)+=1;
			// 	// }
			}
			// m_disparity.at<double>(i,j)+=m_offset;
			m_disparity.at<double>(i,j)*=zoomFactor;
			// double b=thresholdedij.at<double>(i,j);
			// double a=cv::sum(thresholdedij)[0];
			// m_disparity.at<double>(i,j)=a;
		}
	}
	// m_disparity.convertTo(m_disparity, CV_8U);
    imwrite(m_path_to_disparity,m_disparity);
	//cv::namedWindow("disparity Map");
	//cv::imshow("disparity Map", m_disparity);
	// printContentsOf3DCVMat(m_disparity,true,"disparity_map");
	//cv::waitKey(0);
}









































void MatchingAlgorithm::printProperties()
{
	// printf("images of type: ")
	printf ("size_of_images: width (y variable) : %i height (x variable) : %i \n", m_x_size, m_y_size);
	printf("value of mu : %4.2f, tau %4.2f, sigma %4.2f, s %4.2f, Niter %i, m_factor %4.2f, t_size %i and offset %i",m_mu,m_tau,m_sigma,m_s,m_Niter,m_factor,m_t_size,m_offset);
	// m_mu = 75.0/255.0;
	// m_tau = 1.0/sqrt(12.0) ;
	// m_sigma = 1.0/sqrt(12.0) ;
	// m_s=0.3;

	// m_Niter = Niter;//700;
	// m_factor=0.05;
	// m_t_size = t_size ; //need to impose that m_t_size is smaller than m_x_size
	// m_offset=offset;
}



void MatchingAlgorithm::showImages()
{
	cv::namedWindow("Output Image1");
	cv::namedWindow("Output Image2");
	m_image1->convertTo(*m_image1, CV_8U);
	m_image2->convertTo(*m_image2, CV_8U);
	cv::imshow("Output Image1", *m_image1);
	cv::imshow("Output Image2", *m_image2);
	cv::waitKey(0);
}

cv::Mat MatchingAlgorithm::getLayer(cv::Mat Matrix3D,int layer_number)
{
	// 	m_y_size=m_image1->size().height;
	// m_x_size=m_image1->size().width;
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
cv::Mat MatchingAlgorithm::getRow(const cv::Mat &Matrix4D,int numberRow,bool newOne)
// get the  row numer numberRow from a 4D matrix
{
	// if (Matrix4D.dims==4)
	// {
	int dims[] = { Matrix4D.size[1], Matrix4D.size[2],Matrix4D.size[3]};
	if (numberRow > Matrix4D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	cv::Mat extractedMatrix(3,dims, CV_64FC1, Matrix4D.data + Matrix4D.step[0] * numberRow);
	// if (newOne==true)
	// {
	// 	cv::Mat extractedMatrixCloned=extractedMatrix.clone();//(3,dims, CV_64FC1, 0.0);
	// // extractedMatrix.copyTo(extractedMatrixCloned);//(3,dims, CV_64FC1, 0.0);
	// 	return extractedMatrixCloned;
	// }
	// else
	// {
	return extractedMatrix;
	// }

}

cv::Mat MatchingAlgorithm::getRow3D(const cv::Mat &Matrix3D,int numberRow)
// get the  row numer numberRow from a 4D matrix
{
	// if (Matrix4D.dims==4)
	// {
	int dims[] = { Matrix3D.size[1], Matrix3D.size[2]};
	if (numberRow > Matrix3D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	cv::Mat extractedMatrix(2,dims, CV_64FC1, Matrix3D.data + Matrix3D.step[0] * numberRow);
	return extractedMatrix;
}


cv::Mat MatchingAlgorithm::getRow2D(const cv::Mat &Matrix2D,int numberRow)
// get the  row numer numberRow from a 4D matrix
{
	// if (Matrix4D.dims==4)
	// {
	int dims[] = { Matrix2D.size[1]};
	if (numberRow > Matrix2D.size[0] or numberRow < 0)
		{
			throw std::invalid_argument( "received false row" );
		}
	cv::Mat extractedMatrix(1,dims, CV_64FC1, Matrix2D.data + Matrix2D.step[0] * numberRow);
	return extractedMatrix;
}

void MatchingAlgorithm::printContentsOf3DCVMat(const cv::Mat matrix,bool writeOnFile,std::string filename)
{
	if (writeOnFile==false)
	{
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
	// std::cout << "matrix = "<< std::endl << " "  << M << std::endl << std::endl;
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
	// std::cout << "matrix = "<< std::endl << " "  << M << std::endl << std::endl;
	}
	}
	else
	{
	// Declare what you need
		cv::FileStorage file(filename, cv::FileStorage::WRITE);
		// std::ofstream file("FileStorage.txt");
		// if(!file)
    	// {
        	// std::cout<<"File Not Opened"<<std::endl;  return;
    	// }
	// cv::Mat matrix;
		// for (int i=0;i<matrix.size[0];i++)
		// {
		// 	cv::Mat matrixi=getRow3D(matrix,i);
		// 	for (int j=0;j<matrix.size[1];j++)
		// 		{
		// 			cv::Mat matrixij=getRow2D(matrixi,j);
		// 			for (int k=0;k<matrix.size[2];k++)
		// 				{
		// 					file<<"value of pixel"<<i <<" and "<<j<<" and "<<k<<" : "<< matrixij.at<double>(k)<<std::endl;
		// 				}	
		// 		}
		// }
		// file.close();
	// Write to file!
    	// file.writeObj("matrix.txt",&matrix);
    	// file<<"matrixSize"<<5;
		file <<"the matrix"<< matrix;
	}
}

// string type2str(int type) {
//   string r;

//   uchar depth = type & CV_MAT_DEPTH_MASK;
//   uchar chans = 1 + (type >> CV_CN_SHIFT);

//   switch ( depth ) {
//     case CV_8U:  r = "8U"; break;
//     case CV_8S:  r = "8S"; break;
//     case CV_16U: r = "16U"; break;
//     case CV_16S: r = "16S"; break;
//     case CV_32S: r = "32S"; break;
//     case CV_32F: r = "32F"; break;
//     case CV_64F: r = "64F"; break;
//     default:     r = "User"; break;
//   }

//   r += "C";
//   r += (chans+'0');

//   return r;
// }







void MatchingAlgorithm::helpDebug()
{
	std::cout<<"choissisez un entier associé aux objets à afficher : \n 1:contenu image1,\n 2:contenu image2,\n 3:affichage data_term, \n 4:affichage 10 sliceth of v \n 5: contenu terme de données \n 6: test de projCh \n 7: test of projKh\n 8: test of gradh \n 9:testing difference between projKh and projKh_effic 20: quit \n "<<std::endl;
	bool finishTest=false;
	int option;
	std::cin>> option;
	int number;// for the data term

	switch(option)
	{
		case 9:
		{
			std::cout<<"testing difference between projKh et projKh_effic "<<std::endl;
			// cv::Mat v;
				// cv::Mat vproj;
			int size2[4] = {3, m_y_size, m_x_size, m_t_size };
			cv::Mat phi(4, size2, CV_64FC1, 0.0);
			// getRow(phi,0)=cv::Scalar::all(50.0);
			srand (time(NULL));
			// for (int k=0;k<m_t_size;k++)
			// 	{
			// 		v[k]=cv::Mat::ones(m_y_size,m_x_size,CV_64FC1);
			// 		cv::randu(v[k],-3, 3);
			// 	}
			cv::randu(phi,-100.0, 100.0);
			
			clock_t tStartPhiProj_effic = clock();
			cv::Mat phiProj_effic=projKh_effic(phi,m_g);
			printf("Time taken for projKh_effic: %.2fs\n", (double)(clock() - tStartPhiProj_effic)/CLOCKS_PER_SEC);
			clock_t tStartPhiProj = clock();
			cv::Mat phiProj=projKh(phi,m_g);
			printf("Time taken for projKh: %.2fs\n", (double)(clock() - tStartPhiProj)/CLOCKS_PER_SEC);
			cv::Mat result;cv::absdiff(phiProj_effic,phiProj,result);
			std::cout<<cv::sum(result)<<std::endl;
			return;
		}break;
		case 8:
		{
			std::cout<<"test gradh_effic"<<std::endl;
			int size2[3] = {m_y_size, m_x_size, m_t_size };
			cv::Mat v(3, size2, CV_64FC1, 1.0);
			srand (time(NULL));
			cv::randu(v,-100.0, 100.0);
			cv::Mat phi=gradh_effic(v);
			// std::cout<<"choose y,x or t slice :"<<std::endl;
			// int slice; ctd::cin>>slice;
			for (int i=0;i<m_y_size;i++)
				{
					std::cout<<"\n"<<std::endl;
					for (int j=0;j<m_x_size;j++)
						{
							for (int k=0;j<m_t_size;j++)
						{
								// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
							std::cout<<"for pixel"<<i <<" and "<<j<<" and "<<k<<" : "<<"value at (i,j,k) of v"<< v.at<double>(i,j,k)<<"value at (i+1,j,k)"<< v.at<double>(i+1,j,k)<<"value at (i,j+1,k)"<< v.at<double>(i,j+1,k)<<"value at (i,j,k+1)"<< v.at<double>(i,j,k+1)<<std::endl;
							std::cout<<"for the same pixel value of phiy(i,j,k)"<<phi.at<double>(cv::Vec<int,4>(0,i,j,k));
							std::cout<<"for the same pixel value of phix(i,j,k)"<<phi.at<double>(cv::Vec<int,4>(1,i,j,k));
							std::cout<<"for the same pixel value of phit(i,j,k)"<<phi.at<double>(cv::Vec<int,4>(2,i,j,k));
							std::cout<<"\n \n"<<std::endl;
						}
					}
				}

		}break;
		case 1:
		{
			std::cout<<"print contents image1"<<std::endl;
			for (int i=0;i<m_y_size;i++)
				{
					std::cout<<"\n"<<std::endl;
					for (int j=0;j<m_x_size;j++)
						{
							// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
							std::cout<<"value of pixel"<<i <<" and "<<j<<" : "<< m_image1->at<double>(i,j)<<"  "<<std::endl;
						}
				}
				}break;
		case 2:
		{		
			std::cout<<"print contents image2"<<std::endl;
			for (int i=0;i<m_y_size;i++)
				{
					std::cout<<"\n"<<std::endl;
					for (int j=0;j<m_x_size;j++)
						{
								// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
							std::cout<<"value of pixel"<<i <<" and "<<j<<" : "<< m_image2->at<double>(i,j)<<"  "<<std::endl;
						}
				}
		}break;

		case 3:
		{
			std::cout<<"show data term in imageForm, choose a number between : "<<0<<"and tha maximum of dispartity map :"<<m_t_size<<std::endl;
			std::cin>>number;
			cv::namedWindow("data_term");
			// int size3[2] = { m_y_size, m_x_size };
			// cv::Mat g_layer(2, size3, CV_64FC1, 10.0);
			// for (int i=0;i<m_y_size;i++)
			// 	{
			// 	for (int j=0;j<m_x_size;j++)
			// 		{
			// 		// for (int k=0;k<m_t_size,k++)
			// 		// 	{
			// 				g_layer.at<double>(i,j)=m_g.at<double>(i,j,number);
			// 			// }
			// 		}
			// 	}
			cv::Mat g_layer=getLayer(m_g,number);	
			// // cv::Mat m_g[number].copyTo(imageToShow);
			// // m_g[number].convertTo(m_g[number], CV_8U);
			cv::imshow("data_term",g_layer);
			cv::waitKey(0);
		}break;

		// case 4:
		// {
		// 	std::cout<<"show v in imageForm, choose a number between : "<<0<<"and tha maximum of dispartity map :"<<m_t_size<<std::endl;
		// 	std::cin>>number;
		// 	cv::namedWindow("data_term");
		// 	// int size3[2] = { m_y_size, m_x_size };
		// 	// cv::Mat g_layer(2, size3, CV_64FC1, 10.0);
		// 	// for (int i=0;i<m_y_size;i++)
		// 	// 	{
		// 	// 	for (int j=0;j<m_x_size;j++)
		// 	// 		{
		// 	// 		// for (int k=0;k<m_t_size,k++)
		// 	// 		// 	{
		// 	// 				g_layer.at<double>(i,j)=m_g.at<double>(i,j,number);
		// 	// 			// }
		// 	// 		}
		// 	// 	}
		// 	cv::Mat g_layer=getLayer(m_g,number);	
		// 	// // cv::Mat m_g[number].copyTo(imageToShow);
		// 	// // m_g[number].convertTo(m_g[number], CV_8U);
		// 	cv::imshow("data_term",g_layer);
		// 	cv::waitKey(0);
		// }break;
// // double a=m_v.at<double>(10,10,0);
	// // double b=m_v.at<double>(10,10,m_v.size[2]-1);
	// cv::Mat m_vlayerConverted;
	// // (getLayer(m_v,10)).convertTo(m_vlayerConverted, CV_8U);
	// // imwrite("vlayer.jpg",getLayer(m_v,10));

	// cv::namedWindow("Output vlayer10");
	// cv::imshow("Output vlayer10", m_vlayerConverted);
	// cv::waitKey(0);

		case 5:
		{
			std::cout<<"print contents data term, choose a number between : "<<0<<"and tha maximum of dispartity map :"<<m_t_size<<std::endl;
			std::cin>>number;
			cv::Mat g_layer=getLayer(m_g,number);
			for (int i=0;i<m_y_size;i++)
				{
					std::cout<<"\n"<<std::endl;
					for (int j=0;j<m_x_size;j++)
						{
								// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
							std::cout<<"value of pixel"<<i <<" and "<<j<<" : "<< g_layer.at<double>(i,j)<<"  "<<std::endl;
						}
				}
				
			}break;
		case 6:
		{
			std::cout<<"testing projCh "<<std::endl;
			// cv::Mat v;
				// cv::Mat vproj;
			int size1[3] = { m_y_size, m_x_size, m_t_size };
			cv::Mat v(3, size1, CV_64FC1, 1.0);
			srand (time(NULL));
			// for (int k=0;k<m_t_size;k++)
			// 	{
			// 		v[k]=cv::Mat::ones(m_y_size,m_x_size,CV_64FC1);
			// 		cv::randu(v[k],-3, 3);
			// 	}
			cv::randu(v,-3, 3);
			// cv::Mat projv=projCh(v);
			cv::Mat projv_effic=projCh_effic(v);
			// vproj.resize(v.size());
				 // vproj(v,vproj);
			std::cout<<"\n print contents of vproj- before and after projection, choose a number between : "<<0<<" and tha maximum of dispartity map :"<<m_t_size<<"\n"<<std::endl;
			std::cin>>number;
			// cv::Mat projv_layer=getLayer(projv,number);
			for (int i=0;i<m_y_size;i++)
				{
					//std::cout<<"\n"<<std::endl;
					for (int j=0;j<m_x_size;j++)
						{
								// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
							// std::cout<<"value on  pixel "<<i <<" and "<<j<<" of v :"<< v.at<double>(i,j,number)<<" and of vproj "<< projv.at<double>(i,j,number)<<" and of vproj_effic "<< projv_effic.at<double>(i,j,number)<<std::endl;
							std::cout<<"value on  pixel "<<i <<" and "<<j<<" of v :"<< v.at<double>(i,j,number)<<" and of vproj_effic "<< projv_effic.at<double>(i,j,number)<<std::endl;
						}
				}
			}break;
		
		case 7:
		{
			std::cout<<"testing projKh "<<std::endl;
			// cv::Mat v;
				// cv::Mat vproj;
			int size2[4] = {3, m_y_size, m_x_size, m_t_size };
			cv::Mat phi(4, size2, CV_64FC1, 1.0);
			// getRow(phi,0)=cv::Scalar::all(50.0);
			srand (time(NULL));
			// for (int k=0;k<m_t_size;k++)
			// 	{
			// 		v[k]=cv::Mat::ones(m_y_size,m_x_size,CV_64FC1);
			// 		cv::randu(v[k],-3, 3);
			// 	}
			cv::randu(phi,-100.0, 100.0);
			cv::Mat phiProj=projKh(phi,m_g);

			// vproj.resize(v.size());
				 // vproj(v,vproj);
			// int slice;
			// std::cout<<"\n print contents of phi- before and after projection, for y,x or t between : "<<0<<" and  :"<<phiProj.size[0]<<"\n"<<std::endl;
			// std::cin>> slice;

			std::cout<<"\n print contents of phiproj- before and after projection, choose a number between : "<<0<<" and tha maximum of dispartity map :"<<m_t_size<<"\n"<<std::endl;
			
			std::cin>>number;
			for (int i=0;i<m_y_size;i++)
				{
					//std::cout<<"\n"<<std::endl;
					for (int j=0;j<m_x_size;j++)
						{
								// data_out_line_i[j]=m_mu*abs(data_in_1_line_i[j]-data_in_2_line_i[j-k]);
							// m_g[k].at<uchar>(i,j)=m_mu*abs(m_image1->at<uchar>(i,j)-m_image2->at<uchar>(i,j-k+1));
							// std::cout<<"value on  pixel "<<i <<" and "<<j<<" of phi :"<< phi.at<double>(cv::Vec<int,4>(slice,i,j,number))<<" and of phiProj "<< phiProj.at<double>(cv::Vec<int,4>(slice,i,j,number))<<std::endl;
							double phiprojy=phiProj.at<double>(cv::Vec<int,4>(0,i,j,number));
							double phiprojx=phiProj.at<double>(cv::Vec<int,4>(1,i,j,number));
							double normphiprojxy=sqrt(phiprojy*phiprojy+phiprojx*phiprojx);

							double phitplusg=phiProj.at<double>(cv::Vec<int,4>(2,i,j,number))+m_g.at<double>(i,j,number);
							std::cout<<"value on  pixel "<<i <<" and "<<j<<" of norm(phiProjx,phiProjy) "<<normphiprojxy<<" and of phiProjt+g"<<phitplusg<<std::endl;
						}
				}
			}
		case 20:
		{
			finishTest=true;
		}
	}while(!finishTest);
}