#ifndef _PEDDET_H_
#define _PEDDET_H_

#include "vehicleCV.hpp"
#include "debug.hpp"

/////////////////////////////////
// ADAS program API
/////////////////////////////////

long int GetWallTime();

typedef enum {
    LANE_DET = 0,
    CAR_DET_1 = 1,
    CAR_DET_2 = 2,
    CAR_DET_3 = 3,
    CAR_DET_4 = 4,
    PERSON_DET_1 = 5,
    PERSON_DET_2 = 6,
    PERSON_DET_3 = 7,
    PERSON_DET_4 = 8,
    MAX_DET = 5,
}ArithmeticType; //算法类型

typedef struct cPointInfo{
    int x;
    int y;
}PointInfo;

typedef struct cRectInfo{
    PointInfo Top; //顶点
    int Hight;
    int Width;
    float distance;
}RectInfo;

typedef struct cLineInfo{
    PointInfo P1;
    PointInfo P2;
    int attr;   //1为实线 0为虚线
}LineInfo;

/*
typedef struct cCoordinateInfo{ //坐标信息
    int LineNum;
    int CarRectNum;
    int PersonRectNum;
    uchar laneDeparture;
    LineInfo Line[LINE_NUM_MAX];
    RectInfo CarRect[RECT_NUM_MAX];
    RectInfo PerRect[RECT_NUM_MAX];
}CoordinateInfo;
*/

typedef struct cPositionInfo{ //位置信息
    int LineNum;//需要画线条数
    int RectNum;//需要画框个数
    LineInfo Line[LINE_NUM_MAX];//画线详细信息
    RectInfo Rect[RECT_NUM_MAX];//画框详细信息
}PositionInfo;

typedef struct cAlarmInfo{//报警详细信息
    unsigned char laneDeparture;//1向左偏离 0正常 2向右偏离
    unsigned char time; //剩余碰撞时间
    unsigned char distance;//与目标相对距离
    unsigned char targetspeed;//目标速度
    unsigned char suppression;//抑制状态 0：正常 1：抑制
    unsigned char sysstat;//系统状态 0:正常 1：失效 故障
}AlarmInfo;

typedef struct cDetResultInfo{//算法检测结果
    ArithmeticType type;
    PositionInfo position;
    AlarmInfo alarm;
}DetResultInfo;

typedef struct cRecCarInfo{
    unsigned char CarSpeed;
    unsigned short CarSteeringWheel;
    unsigned int CarStat;
}RecCarInfo;

#ifdef ARM
    class ArithmeticBase
    {
    public:
        //ArithmeticBase();
        virtual ~ArithmeticBase(){}
        virtual void InitParam()=0;
        virtual void StopDet()=0;
        virtual DetResultInfo ArithmeticInterFace(unsigned char* pYuvBuf, int height, int width, RecCarInfo* inPtr)=0;
        virtual DetResultInfo* ProcessThreadResult(DetResultInfo* threadresult, int threadNum)=0;
    /*
        其他接口
    */
    };

#else
    class ArithmeticBase
    {
    public:
        ArithmeticBase(){}
        virtual void InitParam(){}
        virtual void StopDet(){}
        // virtual ~ArithmeticBase(){}
        virtual DetResultInfo ArithmeticInterFace(unsigned char* pYuvBuf, int height, int width, RecCarInfo* inPtr){
            DetResultInfo ret;
            return ret;
        }
        virtual DetResultInfo* ProcessThreadResult(DetResultInfo* threadresult, int threadNum){
            return NULL;
        }
        ///其他接口
    };
#endif


//////////////////////////////////
// the interface to use kalmanfilter api
//////////////////////////////////
class Prediction{
    public:
    cv::KalmanFilter kalman;//state(x,y,detaX,detaY)
    cv::Mat process_noise;
    cv::Mat measurement;//measurement(x,y)
    cv::Rect rectMeasPre;
    //cv::Rect rectPre2;
    cv::Rect rectPredict;
    cv::Rect rectTarget;
    int trackID; //indicate the number to count for no dection 
    int detect0Count;
    public:
    //Track() = default;
    //Track(Rect);
    void InitKalman();
    void SetNewTrack(Rect r);
    cv::Rect TrackRect(cv::Rect rectMeas, double rectTID, int predictMode);     
};
//////////////////////////////////
// class to manage multiple tracker
//////////////////////////////////
class Track{
public:
    Prediction Tracker[TRACKER_NUM2];
    cv::Rect rPredict[PREDICT_NUM][TRACKER_NUM];
    cv::Point statePt[PREDICT_NUM][TRACKER_NUM];
    int predictCount;
    int trackCount;
    int nPre;

public:
    void AllocTrack(std::vector<cv::Rect> detedObjs);///allocate tracker to each deteted objects
};


//////////////////////////////////
// frame ROI process function
//////////////////////////////////

class ROI{
    public:
    int scaledHeight;
    int scaledWidth;
    double scaleRate;
    double scaleRatio;
    int roiTop;
    int roiBtm;
    int roiLeft;
    int roiRight;
    int stepSize;
    int rectCount;
    cv::Mat roiImg;
    std::vector<cv::Rect> results;
    //demo peremeters
    int winPosLeft;
    int winPosTop;
    int winNum;
    char winTitle[50];

    public:
    ROI() {rectCount=0;};
    //ROI(int y1, int y2, int x1, int x2, double r): roiTop(y1), roiBtm(y2), roiLeft(x1), roiRight(x2), scaleRatio(r) {}
    cv::Mat ExtractROI (cv::Mat inputImg, int x1, int y1, int x2, int y2, double r);
    void GetDetResult (vector<Rect> detedObjs);
    void PushRect(DetResultInfo& detRsStru, std::vector<Rect>& detRsRect, double rectIdentity );
    void PushRect(DetResultInfo& detRsStru, Rect detRsRect, double rectIdentity );
    void RectNMS(std::vector<Rect> detRsRect);
};

class vehDetect : public ArithmeticBase
{
public:
    long int timeTagPre;
    int threadTag; //indicate different thread
    CvDetection cvVehDet; //modified opencv classifier 
    Track VehTrack; //object for vehicle traction    
    DetResultInfo predictPosResult[PREDICT_NUM]; //the prediction package to return
    cv::Mat frameYUVbkp;
    bool bkpDone;

public:
    vehDetect() = default;
	vehDetect(int t);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
	void InitParam();
   	void StopDet();
    // unsigned char bufYuv[1280*3/2*720];

    DetResultInfo ArithmeticInterFace(unsigned char* pYuvBuf, int height, int width, RecCarInfo* inPtr = NULL);
    DetResultInfo* ProcessThreadResult(DetResultInfo* threadresult, int threadNum);
};

#endif