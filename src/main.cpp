#include <iostream>
#include <string>
#include "DepthFilter.hpp"

using namespace std;
int main(void)
{
	cv::Mat imgD,* imgR=NULL;
	// Gaussian Test

	float sigmae = 3.5;
	float sigmac = 2.5;
	int i, w = 5;
	int start = 112, nimages = 19;
	char buffer[256];

	string name1("image pre process"), name2("image post process");
	DepthFilter::DepthFilter filt(w,sigmae,sigmac);

	// Occlusion filling test
	cv::namedWindow(name1,cv::WINDOW_NORMAL);
	cv::namedWindow(name2,cv::WINDOW_NORMAL);

	for(i=start;i<=start+nimages;i++){
		imgD.release();
		sprintf(buffer,"images/depth_Dani_1_%3i.tiff",i);
		imgD = cv::imread(buffer,cv::IMREAD_GRAYSCALE);
		if(!imgD.data)
		{
			cout<<"Image not available"<<endl;
			return 0;
		}
		imgR = filt.filter(&imgD);
		cv::imshow(name1,imgD);
		cv::imshow(name2,*imgR);
		std::cout<<i<<std::endl;
		cv::waitKey(30);
	}
	std::cout<<"Terminated"<<std::endl;
	return 0;
}
