#ifndef _DBUG_
#define _DBUG_

///////////////
// include set
///////////////
#define DEBUG 1
#define ARM
#define PREDICT_NUM 6
#define TRACKER_NUM 20
#define TRACKER_NUM2 40
#define LINE_NUM_MAX 5
#define RECT_NUM_MAX 100

/////////////////////////////////
// Vehicles Detectioin peremeters
/////////////////////////////////

#define SHOWINFO
// #define SHOWINFO2

// #define DEMA
// #define TRACK_TRACKER
// #define DEMO
#define FRAME_INTERVAL 40 //frame intervel 40ms
#define ROINUM_MAX 10
#define DET0_COUNT_MAX 2 //detect result will remain for DET0_COUNT_MAX*2-2 frames
#define NEW_DET_IGNORE 3

using namespace cv;
using namespace std;

extern long int frameTag;
extern long int timeTag1;
extern long int timeTag2;
extern long int timeTag3;
extern long int timeTag4;
extern long int timeTag5;
extern long int timeTag6;
extern long int timeTag7;
extern long int timeTag8;
extern long int timeTag9;
extern long int timeTag10;
extern long int timeTag11;
extern long int timeTag12;

#define T1 timeTag1 = GetTimeTag();
#define T2 timeTag2 = GetTimeTag();  
#define T4 timeTag4 = GetTimeTag(); 
#define T5 timeTag5 = GetTimeTag(); 
#define T7 timeTag7 = GetTimeTag();
#define T8 timeTag8 = GetTimeTag();
#define T10 timeTag10 = GetTimeTag();
#define T11 timeTag11 = GetTimeTag();

long int GetTimeTag();

#if DEBUG

	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	#include <opencv2/imgproc/imgproc.hpp>
	#include <opencv2/nonfree/nonfree.hpp>
	#include <opencv/cv.h>
	#include <string>
	#include <sstream>
	#include <iostream>
	#include <iomanip>
	#include <stdio.h>
	#include <stdlib.h>
	#include <iostream>
	#include <vector>
	#include <sys/time.h>

	// #define SINGLE_THREAD
	//#define SAVETIMEDATA
	// #define PROCESSPIC
	//#define DRAWROI

	extern int sampNum;
	extern int scaledPos;
	extern int scaledPosStep;

	extern int topOffset;
	extern int btmOffset;
	extern int leftOffset;
	extern int rightOffset;
	extern double scaleOffset;
	extern bool doPause;
	extern bool doShowOri;
	extern bool quit;
	extern bool doShowDet;
	extern bool doSetRoiSize;
	extern bool doBlockResize;
	extern bool doShowSize;
	extern bool doShowRoi;
	extern bool doShowOri;
	extern bool doShowRoiPre;
	extern bool doGetRoiSize;
	extern bool doNMS;
	extern bool doMutiScaleDet;
	extern bool domark;
	extern int debugCount;
	extern int skipRsNum;
	extern std::string classifierName;

	extern double rectID1[50];
	extern double rectID2[50];
	extern double rectID3[50];
	extern double rectID4[50];
	extern int roiNum;
	//extern int winPosTop;
	//extern int winPosLeft;
	//extern char winTitle[50];
	extern bool doMoveWin;
	extern bool doPredict;
	//extern bool doNewTrac;
	extern bool doShowTrack;
	//extern int detect0Count;

	extern pthread_mutex_t mutex;

	//extern std::ofstream tut;

	void MouseEvent(int event, int x, int y, int flags, void* data);
	void GetRectInfo(cv::Mat imgDeted, cv::Mat imgOrig);
	void SaveSample(char* path, cv::Mat originImg, cv::Rect r, cv::Size sampSize, bool doNameInfo = false);

	////////////////////////////
	//   graph cut
	////////////////////////////
	cv::Mat displaySegResult(cv::Mat segments, int numSeg, Mat image = cv::Mat());
	void segMerge(Mat & image, Mat & segments, int & numSeg);
	Mat watershedSegment(Mat & image, int & noOfSegments);
	Mat displaySegResult(Mat  segments, int numOfSegments, Mat  image);

	/////////////////////////
	//  hog descriptor
	/////////////////////////
	std::vector<Mat> cacHOGFeature(cv::Mat srcImage);
	void cacHOGinCell(Mat& HOGCellMat, Rect roi, std::vector<Mat>& integrals);
	cv::Mat getHog(Point pt,std::vector<Mat> &integrals);
	std::vector<Mat> cacHOGFeature(cv::Mat srcImage);

	/////////////////////////
	//  show histogram
	/////////////////////////
	/* ===============  histYUV  ================= */
	void ShowHistYUV(cv::Mat section);
	void ShowHistogram(cv::Mat srcImage);

	/////////////////////////
	// sobel
	/////////////////////////
	cv::Mat GetSobel(cv::Mat srcImage);
	void ShowNewSobel(cv::Mat srcGray);
	cv::Mat roberts(cv::Mat srcImage); 
	cv::Mat MarrEdge(cv::Mat src);
#endif

#endif