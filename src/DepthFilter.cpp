/*
 * DepthFilter.cpp
 *
 *  Created on: Aug 8, 2013
 *      Author: Jose Juan Hernandez Lopez
 */

#include "../include/DepthFilter/DepthFilter.hpp"

namespace DepthFilter {

DepthFilter::DepthFilter(int ws, float sigmae, float sigmas)
:ws(ws), sigmae(sigmae), sigmas(sigmas){
	gaussW = calculate_gauss2D(ws, sigmae);
}

DepthFilter::~DepthFilter() {
	delete [] gaussW;
}

void DepthFilter::pointProjection(Mat & img, KLT_FeatureList fl){

	int ind = 0,i,j,x,y,rows,cols,k = 0,it, itr;
	float aux = (sigmas*sigmas)/2;
	float numx,denx;
	float numy,deny;
	float numz,denz;
	float tmp;
	rows = img.rows;
	cols = img.cols;

	float * ptr = img.ptr<float>();
	while(true){
		if(ind>=fl->nFeatures)
			break;

		if(fl->feature[ind]->x == -1 || fl->feature[ind]->y == -1){
			fl->feature[ind]->_3Dx = -1;
			fl->feature[ind]->_3Dy = -1;
			fl->feature[ind]->_3Dz = -1;
			fl->feature[ind]->_3Dlost = 0;
		}
		else
		{
			y = (int)round(fl->feature[ind]->y);
			x = (int)round(fl->feature[ind]->x);
			k = 0;
			numx = 0;
			denx = 0;
			numy = 0;
			deny = 0;
			numz = 0;
			denz = 0;

			//cout<< "x: "<<x<<"    y: "<<y<<endl;
			itr = 3*(y*cols+x);
			for(i = std::max(y-ws/2,0); i <= std::min(y+ws/2,rows-1); i++,k++ )
				for(j = std::max(x-ws/2,0); j<= std::min(x+ws/2,cols-1); j++){
					it = 3*(i*cols+j);
					//std::cout<<std::setw(8)<< j <<","<<i<<"  "<<it<<"    " << ptr[it]<<"     ";
					if(ptr[it]!=0)
					{
						tmp = gaussW[k]*exp(-((ptr[it]-ptr[itr])*(ptr[it]-ptr[itr]))/aux);
						//cout <<setw(8)<<gaussW[k]<<"  "<<exp(-((ptr[it]-ptr[itr])*(ptr[it]-ptr[itr]))/aux)<<"     "<<tmp;
						numx += ptr[it]*tmp;
						denx += tmp;
					}
					if(ptr[it+1]!=0)
					{
						tmp = gaussW[k]*exp(-((ptr[it]-ptr[itr+1])*(ptr[it]-ptr[itr+1]))/aux);
						numy += ptr[it+1]*tmp;
						deny += tmp;
					}
					if(ptr[it+2]!=0)
					{
						tmp = gaussW[k]*exp(-((ptr[it]-ptr[itr+2])*(ptr[it]-ptr[itr+2]))/aux);
						numz += ptr[it+2]*tmp;
						denz += tmp;
					}
					//cout<<endl;
				}


			if((denx != 0) && (deny != 0) && (denz != 0)){
				fl->feature[ind]->_3Dx = numx/denx;
				fl->feature[ind]->_3Dy = numy/deny;
				fl->feature[ind]->_3Dz = numz/denz;
				fl->feature[ind]->_3Dlost = 0;
			}
			else {
				fl->feature[ind]->_3Dx = 0;
				fl->feature[ind]->_3Dy = 0;
				fl->feature[ind]->_3Dz = 0;
				fl->feature[ind]->_3Dlost = 1;
			}
		}
		ind++;
	}
}

void DepthFilter::StereoPointProjection(Mat & img, KLT_FeatureList fl, Mat & Q){
	int ind = 0,i,j,x,y,rows,cols,k = 0,it, itr;
	float aux = (sigmas*sigmas)/2;
	float num,den;

	float tmp;
	double* temp2;
	rows = img.rows;
	cols = img.cols;

	float * ptr = img.ptr<float>();
	while(true){
		if(ind>=fl->nFeatures)
			break;

		if(fl->feature[ind]->x == -1 || fl->feature[ind]->y == -1){
			fl->feature[ind]->_3Dx = -1;
			fl->feature[ind]->_3Dy = -1;
			fl->feature[ind]->_3Dz = -1;
			fl->feature[ind]->_3Dlost = 0;
		}
		else
		{
			y = (int)round(fl->feature[ind]->y);
			x = (int)round(fl->feature[ind]->x);
			k = 0;
			num = 0;
			den = 0;

			//cout<< "x: "<<x<<"    y: "<<y<<endl;
			itr = y*cols+x;
			for(i = std::max(y-ws/2,0); i <= std::min(y+ws/2,rows-1); i++,k++ )
				for(j = std::max(x-ws/2,0); j<= std::min(x+ws/2,cols-1); j++){
					it = i*cols+j;
					//std::cout<<std::setw(8)<< j <<","<<i<<"  "<<it<<"    " << ptr[it]<<"     ";
					if(ptr[it]>0)
					{
						tmp = gaussW[k]*exp(-((ptr[it]-ptr[itr])*(ptr[it]-ptr[itr]))/aux);
						//cout <<setw(8)<<gaussW[k]<<"  "<<exp(-((ptr[it]-ptr[itr])*(ptr[it]-ptr[itr]))/aux)<<"     "<<tmp;
						num += ptr[it]*tmp;
						den += tmp;
					}
					//cout<<endl;
				}

			if(den != 0){
				cv::Mat_<double> vec(4,1);
				vec(0)=x; vec(1)=y; vec(2)=num/den; vec(3)=1;
				temp2 = Q.ptr<double>(0);
				vec = Q*vec;
				vec /= vec(3);
				fl->feature[ind]->_3Dx = vec(0)/100;
				fl->feature[ind]->_3Dy = vec(1)/100;
				fl->feature[ind]->_3Dz = vec(2)/100;
				fl->feature[ind]->_3Dlost = 0;
			}
			else {
				fl->feature[ind]->_3Dx = 0;
				fl->feature[ind]->_3Dy = 0;
				fl->feature[ind]->_3Dz = 0;
				fl->feature[ind]->_3Dlost = 1;
			}
		}
		ind++;
	}
}

/*
 * Spatial occlusion filling
 */
void DepthFilter::gaussfilt(cv::Mat * imgD, cv::Mat * imgR){
	// Variables
	float num(0), den(0);

	*(imgR) = imgD->clone();
	int w = this->ws;
	int ws = w/2, i ,j, x, y;
	int nr = imgR->rows, nc = imgR->cols;
	// Point detection occlusions and faults

	for(i=ws;i<nr-ws;i++)
		for(j=ws;j<=nc-ws;j++)
			if(imgD->data[i*nc+j]==0){
				num = 0;
				den = 0;
				for(y = -ws; y<= ws; y++)
					for(x = -ws; x<= ws; x++){
						//std::cout<<std::setw(8)<<(i+y)*nc+(j+x);
						if(imgD->data[(i+y)*nc+(j+x)]!=0)
						{
							num += imgD->data[(i+y)*nc+(j+x)]*gaussW[(y+ws)*w+(x+ws)];
							den += gaussW[(y+ws)*w+(x+ws)];
						}
					}
				if(den != 0){
					imgR->data[i*nc+j] = (char)(num/den);
					//std::cout<<"..";
				}
				//std::cout<<std::endl;
			}
}


/*
 * Crates gaussian window, Memory should be liberated manually
 */
float * DepthFilter::calculate_gauss2D(int w, float sigma){
	float * gauss2D = new float[w*w];
	int i, j, k=0;
	for(i=-w/2;i<=w/2;i++)
		for(j=-w/2;j<=w/2;j++)
			gauss2D[k++] = exp(-(float)(i*i+j*j)/(sigma*sigma)/2);
	return gauss2D;
}

/*
 * Crates a gaussian vector, Memory should be liberated manually
 */
float * DepthFilter::calculate_gauss(int w, float sigma){
	float * gauss = new float[w];
	int i, k=0;
	for(i=0;i<w;i++)
		gauss[k++] = exp(-(float)(i*i)/(sigma*sigma)/2);
	return gauss;
}



} /* namespace DepthFilter */
