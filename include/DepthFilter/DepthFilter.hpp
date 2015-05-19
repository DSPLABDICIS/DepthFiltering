/*
 * DepthFilter.h
 *
 *  Created on: Aug 8, 2013
 *      Author: Jose Juan Hernandez Lopez
 */

#ifndef DEPTHFILTER_H_
#define DEPTHFILTER_H_
#include <iostream>
#include <algorithm>
#include <cmath>
#include "cv.h"
#include "highgui.h"
#include "klt.h"


namespace DepthFilter {

using namespace std;
using namespace cv;
// This class manages inside memory. Output images are deleted by the class.
class DepthFilter {
public:
	/*
	 * n : number of images in time filtering.
	 * ws : Spatial window size.
	 * sigmae : Spatial Sigma.
	 * sigmac : Temporal filter time Sigma.
	 * sigmas : Temporal filter value Sigma.
	 */
	DepthFilter(int ws, float sigmae, float sigmas);
	virtual ~DepthFilter();

	/*Occlusion filling and temporal stabilization if nImages images or more*/
	cv::Mat* filter(cv::Mat* imgD);
	void pointProjection(Mat & img, KLT_FeatureList fl);
	void StereoPointProjection(Mat & img, KLT_FeatureList fl, Mat & Q);

private:
	int ws;						//Window size on gaussian filtering
	int fs;						//
	float sigmae;				//Sigma for gaussian filtering
	float sigmas;				//Sigma for temporal filtering (value gaussian)
	float *gaussW;				//Gaussian window ws*ws

	float * calculate_gauss2D(int w, float sigma);
	float * calculate_gauss(int w, float sigma);
	void gaussfilt(cv::Mat * imgD, Mat * imgR);
};

} /* namespace DepthFilter */
#endif /* DEPTHFILTER_H_ */
