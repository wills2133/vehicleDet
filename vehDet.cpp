#include "vehDet.hpp"

// The detector

/////////////////////
// detect function
/////////////////////




char version[] = "1.33"; ///CES excibition2

///calculate yuv color histogram
#if 0
void CalcHistYUV(cv::Mat sectionYUV, cv::Rect objRect)
{
    int nWidth = 1280, nHeight = 720;
    memset(sectionYUV.data, 128, nHeight*nWidth*sizeof(unsigned char));
    Mat imgRGB;
    cv::cvtColor(sectionYUV, imgRGB, CV_YUV2BGR_I420);

    Mat sectionRGB;
    imgRGB(objRect).copyTo(sectionRGB);
    imshow("sectRGB", sectionRGB);

    // objRect.width = (floor(objRect.width/4))*4;
    // objRect.height = (floor(objRect.height/4))*4;

    
    // cout<<"testImg.cols "<<testImg.cols<<" testImg.rows "<<testImg.rows<<endl;

    long int imgHeadU = nWidth*nHeight;
    long int imgHeadV = nWidth*nHeight*5/4;

    cv::Mat plateU(nHeight/2, nWidth/2, CV_8UC1); 
    cv::Mat plateV(nHeight/2, nWidth/2, CV_8UC1);

    plateU.data = sectionYUV.data + imgHeadU;
    plateV.data = sectionYUV.data + imgHeadV;

    // imshow("plateU", plateU);
    // imshow("plateV", plateV);
 
    cv::Rect objRectU(objRect.x/2, objRect.y/2, objRect.width/2, objRect.height/2);
    cv::Rect objRectV(objRect.x/2, objRect.y/2, objRect.width/2, objRect.height/2);


    Mat objSectU, objSectV;
    plateU(objRectU).copyTo(objSectU);
    plateV(objRectV).copyTo(objSectV);

    imshow("objSectU", objSectU);
    imshow("objSectV", objSectV);

    // unsigned char yuvPixels[objRect.x*objRect.y*3/2];
    // memset(yuvPixels, 128, objRect.x*objRect.y*sizeof(unsigned char));
    // memcpy(yuvPixels+objRect.x*objRect.y*sizeof(unsigned char), objSectU.data, objSectU.cols*objSectU.rows*sizeof(unsigned char));
    // memcpy(yuvPixels+objRect.x*objRect.y*sizeof(unsigned char)*5/4, objSectV.data, objSectV.cols*objSectV.rows*sizeof(unsigned char));

    // Mat sectYUV(objRect.y*3/2, objRect.x, CV_8UC1);
    // sectYUV.data = yuvPixels;
    // Mat imgRGB;
    // cv::cvtColor(sectYUV, imgRGB, CV_YUV2BGR_I420);
    // imshow("sectRGB", imgRGB);


    waitKey(0);

    // 初始化直方图计算参数
    Mat plateYUV[] = { plateU, plateV };
    // plateYUV = &plateU;
    // plateYUV[1] = &plateV;
    int histSizeU = 16;
    int histSizeV = 16;
    int histSize[] = {histSizeU, histSizeV}; 
    float rangeUmin = 0, rangeUmax = 256;
    float rangeVmin = 0, rangeVmax = 256;
    float rangeU[] = { rangeVmin, rangeUmax }; 
    float rangeV[] = { rangeVmin, rangeVmax }; 
    const float* histRange[] = { rangeU, rangeV }; 
    bool uniform = true; 
    bool accumulate = false;
    cv::Mat hist;
    // 计算各个通道的直方图
    calcHist( plateYUV, 2, 0, cv::Mat(), hist, 2, 
        histSize, histRange, uniform, accumulate );

    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);

    // int scale = 10;
    Mat histImg = Mat::zeros(512, histSizeU*histSizeV*5, CV_8UC3);
    cout<<"histImg size"<<histImg.size()<<endl;
    //draw color bar
    for(int v=0; v<histSizeV; v++){
        for(int u=0; u<histSizeU; u++){
        
            int k = v*histSizeU + u;
            // int k = u*histSizeV + v;
            float binVal = hist.at<float>(u, v);
            int intensity = cvRound(binVal*512/maxVal);
            // binClr = CvScalar

            Mat clrSampleYUV(30, 20, CV_8UC1);
            unsigned char clrPixels[30*20];
            memset(clrPixels, 128, sizeof(unsigned char)*400);
            memset(clrPixels+sizeof(unsigned char)*400, u*(rangeUmax-rangeUmin)/histSizeU+rangeUmin, sizeof(unsigned char)*100);
            memset(clrPixels+sizeof(unsigned char)*500, v*(rangeVmax-rangeVmin)/histSizeV+rangeVmin, sizeof(unsigned char)*100);
            // memset(clrPixels+sizeof(unsigned char)*400, 100, sizeof(unsigned char)*100);
            // memset(clrPixels+sizeof(unsigned char)*500, 100 , sizeof(unsigned char)*100);
            clrSampleYUV.data = clrPixels;
            Mat clrSampleRGB;
            cv::cvtColor(clrSampleYUV, clrSampleRGB, CV_YUV2BGR_I420);
            rectangle( histImg, Point( k*5, 512-intensity ), Point( (k+1)*5, 512 ), mean(clrSampleRGB), CV_FILLED );
            int histNum = u*(rangeUmax-rangeUmin)/histSizeU+rangeUmin;
            int histNum2 = v*(rangeVmax-rangeVmin)/histSizeV+rangeVmin;
            if(intensity>0){
                cout<<"u "<<histNum<<" v "<<histNum2<<" intensity "<<intensity<<endl; 
            }
            if(k%10==0){
                char charText[100];
                sprintf(charText, "%d", histNum);
                cv::putText(histImg, charText, Point( k*5+5, 512-intensity-10 ), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 1);
                char charText2[100];
                sprintf(charText2, "%d", histNum2);
                cv::putText(histImg, charText2, Point( k*5+5, 512-intensity  ), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 1);
            }
            // imshow("clrSampleRGB", clrSampleRGB);
            // imshow("calcHist", histImg );
            // waitKey(0);
        }
    }


    imshow("calcObjHist", histImg );
    waitKey(0);

}
#endif 

///get current time
long int GetWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return time.tv_usec/1000 + time.tv_sec*1000;  
}

///extract region of interest from source image
cv::Mat ROI::ExtractROI (cv::Mat inputImg, int x1, int y1, int x2, int y2, double r)
{
    #if DEBUG
        if(doSetRoiSize == false)
    #endif
        {
            roiTop = y1;
            roiBtm = y2;
            roiLeft = x1;
            roiRight = x2;
        }

    ///extrace roi img
    roiImg = cv::Mat(inputImg.rows, inputImg.cols, inputImg.type());           
    inputImg(cv::Rect(roiLeft,roiTop,roiRight-roiLeft,roiBtm-roiTop)).copyTo(roiImg);

    scaleRate = r;

    #if DEBUG
        if(doShowSize){    ///show roi window
            pthread_mutex_lock(&mutex);
            //std::cout<<"inputImg->height "<<scaledHeight<<" inputImg->width "<<scaledWidth<<std::endl;
            std::cout<<"roiTop = "<<roiTop<<"; roiBtm = "<<roiBtm<<"; roiLeft = "<<roiLeft<<"; roiRight = "<<roiRight<<"; scaleRatio = "<<scaleRate<<";"<<std::endl;
            std::cout<<"topOffset "<<topOffset<<" btmOffset "<<btmOffset<<" leftOffset "<<leftOffset<<" rightOffset "<<rightOffset<<"\n "<<std::endl;
            pthread_mutex_unlock(&mutex);
        }
    #endif

    return roiImg;
}




///return the position in th origin frame
void ROI::GetDetResult (vector<Rect> detedObjs)
{
    ///calculate the position on the original image
    for(unsigned int i=0;i<detedObjs.size();i++){
        //if(resultRoi->position.RectNum<RECT_NUM_MAX)
        {   
            cv::Rect rectTmp;
            rectTmp.x = detedObjs[i].x + roiLeft;
            rectTmp.y = detedObjs[i].y + roiTop;
            rectTmp.width = detedObjs[i].width;
            rectTmp.height = detedObjs[i].height;
            results.push_back(rectTmp);   
        //std::cout<<"detedObjs[i].x "<<detedObjs[i].x<<" detedObjs[i].y "<<detedObjs[i].y<<" detedObjs[i].width "<<detedObjs[i].width<<" detedObjs[i].height "<<detedObjs[i].height<<std::endl;
        }
    }

    #if DEBUG
        //show info by each roi
        if(doShowRoi){

            char xscaleLable1[10];
            sprintf(xscaleLable1, "X%.2f", scaleRate);
            cv::putText(roiImg, xscaleLable1, cv::Point(10,15), 1, 1, CV_RGB(0,0,0), 2);
            char xscaleLable2[10];
            sprintf(xscaleLable2, "X%.2f", scaleRate);
            cv::putText(roiImg, xscaleLable2, cv::Point(10,30), 1, 1, CV_RGB(255,255,255), 2);
            cv::rectangle(roiImg,cvPoint(5,5),cvPoint(5+36/scaleRate, 5+36/scaleRate),CV_RGB(255,250,250), 1);

            for(unsigned int i=0;i<detedObjs.size();i++){
                ///draw original ret on roiImage
                cv::rectangle(roiImg,cvPoint(detedObjs[i].x,detedObjs[i].y),cvPoint(detedObjs[i].x+detedObjs[i].width, detedObjs[i].y+detedObjs[i].height),CV_RGB(255,250,250), 1);
                double scanWinWidth = 36, finalWidth  = detedObjs[i].width;
                double scaledRate = scanWinWidth/finalWidth;
                char scaleLabel[10];
                sprintf(scaleLabel, "%.2f", scaledRate);
                cv::putText(roiImg, scaleLabel, cv::Point(detedObjs[i].x+detedObjs[i].width,detedObjs[i].y+detedObjs[i].height+i*2), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255,250,250), 1);
                //imshow("roi", roiImg);
                //cvWaitKey(1);
            }   

            pthread_mutex_lock(&mutex);
            memset(winTitle, 0, sizeof(winTitle));
            sprintf(winTitle, "detRoi_%d", winNum);
            cv::imshow(winTitle, roiImg);
            if(doMoveWin)
            cv::moveWindow(winTitle, winPosLeft, winPosTop);
            pthread_mutex_unlock(&mutex);
        }
    #endif
}


///save detect rectangulars in the structure interface retrun to main()
void ROI::PushRect(DetResultInfo& detRsStru, std::vector<Rect>& detRsRect, double rectIdentity)
{
    for(unsigned int i=detRsStru.position.RectNum; i<(detRsStru.position.RectNum+detRsRect.size()); i++){
        detRsStru.position.Rect[i].Top.x = detRsRect[i].x ;
        detRsStru.position.Rect[i].Top.y = detRsRect[i].y ;
        detRsStru.position.Rect[i].Width = detRsRect[i].width;
        detRsStru.position.Rect[i].Hight = detRsRect[i].height;
    }
    detRsStru.position.RectNum+=detRsRect.size();
    rectCount=detRsStru.position.RectNum;
}

///save detect rectangulars in the structure interface retrun to main()
void ROI::PushRect(DetResultInfo& detRsStru, Rect detRsRect, double rectIdentity)
{
    detRsStru.position.Rect[detRsStru.position.RectNum].Top.x = detRsRect.x ;
    detRsStru.position.Rect[detRsStru.position.RectNum].Top.y = detRsRect.y ;
    detRsStru.position.Rect[detRsStru.position.RectNum].Width = detRsRect.width;
    detRsStru.position.Rect[detRsStru.position.RectNum].Hight = detRsRect.height;
    detRsStru.position.RectNum++;
    //rectID[rectCount] = rectIdentity;
    //cout<<"detRsStru.position.Rect[rectCount].Top.x "<<detRsStru.position.Rect[rectCount].Top.x<<" rectCount "<<rectCount<<" rectID[rectCount] "<<rectID[rectCount]<<endl;
}

/*
void Track::Track(Rect r){
    centerMeasPre = Point(r.x+r.width/2,r.y+r.height/2);
    kalman.statePost.ptr<float>(0)[0] = centerMeasPre.x;
    kalman.statePost.ptr<float>(1)[0] = centerMeasPre.y;
    Point statePost( kalman.statePost.ptr<float>(0)[0], kalman.statePost.ptr<float>(1)[0]);
    cout<<"statePostx "<<statePost.x<<" statePosty "<<statePost.y<<endl;
}
*/

/// reset the track data for a new detect object
void Prediction::SetNewTrack(cv::Rect r)
{
    rectPredict = r;
    rectMeasPre = r;
    kalman.statePost.ptr<float>(0)[0] = r.x+r.width/2;
    kalman.statePost.ptr<float>(1)[0] = r.y+r.height/2;
    Point statePost( kalman.statePost.ptr<float>(0)[0], kalman.statePost.ptr<float>(1)[0]);

    // rectMeasPre.width = rectMeasPre.height = 0;
    // rectMeasPre.x = rectMeasPre.y = 0;
    //cout<<"statePostx "<<statePost.x<<" statePosty "<<statePost.y<<endl;
}



/// initialize a new kalman tracker
void Prediction::InitKalman()
{   
    rectMeasPre = Rect();
    rectPredict = Rect();
    //1.kalman filter setup
    const int stateNum=4;
    const int measureNum=2;
    //initialize kalmanfilter inner parameters
    kalman = KalmanFilter( stateNum, measureNum, 0 );//state(x,y,detaX,detaY)
    process_noise = Mat( stateNum, 1, CV_32FC1 );
    measurement = Mat( measureNum, 1, CV_32FC1 );//measurement(x,y)
    //set transitionmatrix    
    kalman.transitionMatrix = (cv::Mat_<float>(4,4)<<
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1);

    cv::setIdentity(kalman.measurementMatrix, cv::Scalar::all(1) );
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-5));
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));
    //initial predict point
    kalman.statePost.ptr<float>(0)[0] = 640;
    kalman.statePost.ptr<float>(1)[0] = 360;

}


///track by predicting function according to the current positong and the object ID
cv::Rect Prediction::TrackRect(cv::Rect rectMeas, double rectTID, int predictMode)//use pre-predit point as direct empty predict output
{
    Point centerPrdt = Point( rectPredict.x + rectPredict.width/2, rectPredict.y + rectPredict.height/2 );
    // cout<<"Pred1 x "<<centerPrdt.x<<" y "<<centerPrdt.y<<" w "<<rectPredict.width<<endl;
    Point centerMeasPre = Point( rectMeasPre.x + rectMeasPre.width/2, rectMeasPre.y + rectMeasPre.height/2 );
    // cout<<"Pre1 x "<<centerMeasPre.x<<" y "<<centerMeasPre.y<<" w "<<rectMeasPre.width<<endl;
    Point centerMeas( rectMeas.x + rectMeas.width/2, rectMeas.y + rectMeas.height/2 );
    // cout<<"Meas1.x "<<centerMeas.x<<" Meas1.y "<<centerMeas.y<<endl;
    Point statePost( kalman.statePost.ptr<float>(0)[0], kalman.statePost.ptr<float>(1)[0]);
    // cout<<"state1.x "<< statePost.x<<" state1.y "<<statePost.y<<endl; 
    Point ptPredict;
    Mat prediction;

    //smooth the rectsize between roi
    if(rectTID>1&&rectTID<4)
    {
        // cout<<"Pre1-Pred1 "<<rectMeasPre.width-rectPredict.width<<" vs "<<rectPredict.width*0.1<<endl;
        // cout<<"Pre1-Pred1 "<<rectPredict.width-rectMeasPre.width<<" vs "<<rectPredict.width*0.1<<endl;
        if( rectMeasPre.width-rectPredict.width >= rectPredict.width*0.1 ){
            // printf("mark1\n");
            rectPredict.width = rectPredict.height = rectPredict.width + (rectPredict.width*0.05);
            // cout<<"rectPredict.width "<<rectPredict.width<<endl;
        }
        else if( rectPredict.width-rectMeasPre.width >= rectPredict.width*0.1 ){
            // printf("mark2\n");
            rectPredict.width = rectPredict.height = rectPredict.width - (rectPredict.width*0.05);
            // cout<<"rectPredict.width "<<rectPredict.width<<endl;
        }
    }

    int count = 0;
    //cout<<"statePostx2 "<<statePost.x<<" statePosty2 "<<statePost.y<<endl;
    if(predictMode < 0){
        // cout<<"rpredictMode "<<predictMode<<endl;
        // cout<<"  debugCount "<<debugCount<<endl;
        //while(abs(centerMeas.x-kalman.statePost.ptr<float>(0)[0])>10||abs(centerMeas.y-kalman.statePost.ptr<float>(1)[0])>10)
        {
            // cout<<"~~no."<<count<<" absx "<<abs(centerPrdt.x-centerMeas.x)<<endl;
            // cout<<" centerPrdt.x "<<centerPrdt.x<<" centerMeas.x "<<centerMeas.x<<endl;
            // cout<<"count "<<count<<endl;
            count++;
            if(count>10) {
                count=0; 
                // break;
            }
            #if DEBUG
                if(count>2) debugCount++;
            #endif
            
            
            //2.kalman prediction
            prediction = kalman.predict();
            
            ptPredict = cvPoint((int)prediction.ptr<float>(0)[0],(int)prediction.ptr<float>(1)[0]);
            //cout<<"ptprdt0.x "<<ptPredict.x<<" ptprdt0.y "<<ptPredict.y<<endl;

            // cout<<"state10.x "<< kalman.statePost.ptr<float>(0)[0]<<" state10.y "<<kalman.statePost.ptr<float>(1)[0]<<endl; 
            //3.update measurement
            int residx = centerMeas.x-kalman.statePost.ptr<float>(0)[0], residy = centerMeas.y-kalman.statePost.ptr<float>(1)[0];
            ///residual predict use the different distance from last positon
            ///need a boundary check
            if(rectTID<0){
                // printf("residual predict\n");
                measurement.ptr<float>(0)[0]=(float)centerMeas.x+3*residx;
                measurement.ptr<float>(1)[0]=(float)centerMeas.y+3*residy;
            }else{
                measurement.ptr<float>(0)[0]=(float)centerMeas.x;
                measurement.ptr<float>(1)[0]=(float)centerMeas.y;
                // if((float)centerMeas.x+centerMeas.x-rectPredict.x>0&&(float)centerMeas.x+centerMeas.x-rectPredict.x<1280)
                //     measurement.ptr<float>(0)[0]=(float)centerMeas.x+centerMeas.x-rectPredict.x;
                // else
                //     measurement.ptr<float>(0)[0]=(float)centerMeas.x;
                // if(centerMeas.y+centerMeas.y-rectPredict.y>0&&centerMeas.y+centerMeas.y-rectPredict.y<720)
                //     measurement.ptr<float>(1)[0]=(float)centerMeas.y+centerMeas.y-rectPredict.y;
                // else
                //     measurement.ptr<float>(1)[0]=(float)centerMeas.y;
            }
            //4.update
            kalman.correct( measurement );
            //save for next predection
            //centerPrdt = ptPredict;
            centerPrdt = Point(kalman.statePost.ptr<float>(0)[0], kalman.statePost.ptr<float>(1)[0]);
            //centerPrdt = Point((int)prediction.ptr<float>(0)[0], (int)prediction.ptr<float>(1)[0]);
            rectPredict.x = centerPrdt.x-rectPredict.width/2;
            rectPredict.y = centerPrdt.y-rectPredict.height/2;
            // cout<<"Pre0.x "<<rectMeasPre.x + rectMeasPre.width/2<<" Pre0.y "<<rectMeasPre.y + rectMeasPre.height/2<<endl;
            // cout<<"Pre0.x "<<centerMeasPre.x<<" Pre0.y "<<centerMeasPre.y<<endl;
            rectMeasPre = rectMeas;
            //rectMeasPre = Point();
            // cout<<"Meas0.x "<<rectMeas.x + rectMeas.width/2<<" Meas0.y "<<rectMeas.y + rectMeas.height/2<<endl;
            // cout<<"Mfix0.x "<<measurement.ptr<float>(0)[0]<<" Mfix0.y "<<measurement.ptr<float>(1)[0]<<endl;
            // cout<<"state0.x "<< kalman.statePost.ptr<float>(0)[0]<<" state1.y "<<kalman.statePost.ptr<float>(1)[0]<<endl;
            // cout<<"centpd.x "<< centerPrdt.x<<" centpd.y "<<centerPrdt.y<<endl;
            // cout<<"pred0.x "<<(int)prediction.ptr<float>(0)[0]<<" pred0.y "<<(int)prediction.ptr<float>(1)[0]<<endl;

        }
    }else if(predictMode == 1){
        // cout<<"predictMode "<<predictMode<<endl;
        //2.kalman prediction
        prediction = kalman.predict();
        ptPredict = cvPoint((int)prediction.ptr<float>(0)[0],(int)prediction.ptr<float>(1)[0]);
        rectPredict.x = ptPredict.x-rectPredict.width/2;
        rectPredict.y = ptPredict.y-rectPredict.height/2;
        //cout<<"ptPredict.x "<<ptPredict.x<<"ptPredict.y "<<ptPredict.y<<endl;   
    }else if(predictMode == 2){//use pre-measurement point as direct empty predict output
        // cout<<"predictMode "<<predictMode<<endl;
        rectPredict = rectMeasPre;
    }else if(predictMode == 3){//use last predict point for empty predition
        // cout<<"predictMode "<<predictMode<<endl;
        rectPredict = rectPredict;//keep the las predict
    }else if(predictMode == 4){//use pre-measurement point as predition measurement input
        // cout<<"predictMode "<<predictMode<<endl;
        //2.kalman prediction
        prediction = kalman.predict();
        ptPredict = cvPoint((int)prediction.ptr<float>(0)[0],(int)prediction.ptr<float>(1)[0]);
            
        //3.update measurement
        measurement.ptr<float>(0)[0]=(float)centerMeasPre.x;
        measurement.ptr<float>(1)[0]=(float)centerMeasPre.y;

        //4.update
        kalman.correct( measurement );
        rectPredict.x =  ptPredict.x-rectPredict.width/2;
        rectPredict.y =  ptPredict.y-rectPredict.height/2;
        //cout<<"ptPredict.x "<<ptPredict.x<<"ptPredict.y "<<ptPredict.y<<endl;  
    }else if(predictMode == 5){//use pre-predict point as predition measurement input
        //cout<<"predictMode "<<predictMode<<endl;
        //2.kalman prediction
        prediction = kalman.predict();
        ptPredict = cvPoint((int)prediction.ptr<float>(0)[0],(int)prediction.ptr<float>(1)[0]);
            
        //3.update measurement 
        measurement.ptr<float>(0)[0]=(float)centerMeas.x;//+5*(centerMeas.x-centerMeasPre.x);
        measurement.ptr<float>(1)[0]=(float)centerMeas.y;//+5*(centerMeas.y-centerMeasPre.y);

        //4.update
        kalman.correct( measurement );
        rectPredict.x = kalman.statePost.ptr<float>(0)[0]-rectPredict.width/2;
        rectPredict.y = kalman.statePost.ptr<float>(0)[1]-rectPredict.height/2;
        //cout<<"ptPredict.x "<<ptPredict.x<<"ptPredict.y "<<ptPredict.y<<endl;   
    }


    // cout<<"Meas.x "<<rectMeasPre.x + rectMeasPre.width/2<<" Meas.y "<<rectMeasPre.y + rectMeasPre.height/2<<endl;
    // cout<<"Measfix.x "<<measurement.ptr<float>(0)[0]<<" Measfix.y "<<measurement.ptr<float>(1)[0]<<endl;
    // cout<<"statePost.x "<< statePost.x<<" statePost.y "<<statePost.y<<endl; 
    // cout<<"rPrdt.x "<<rectPredict.x+rectPredict.width/2<<" rPrdt.y "<<rectPredict.y+rectPredict.height/2<<endl<<endl;
    // if(rectPredict.x+rectPredict.width/2<1)
    //     waitKey(0);

    return rectPredict;
} 


///allocate tracker to each deteted objects
void Track::AllocTrack(std::vector<cv::Rect> detedObjs)
{   

    int rectIDtmp[100];
    int detedObjsNum = detedObjs.size();//save number of detected objects
    //cout<<"predictCount"<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<predictCount<<endl;
    //cout<<"detedObjsNum1 = "<<detedObjsNum<<endl;
    // for (int i = 0; i < TRACKER_NUM; ++i){
    // cout<< "T"<<i<<" "<< Tracker[i].trackID<<" C "<<Tracker[i].detect0Count<<" x "<<Tracker[i].rectTarget.x<<" w "<<Tracker[i].rectTarget.width<<endl;
    // }
    for(int j=0; j<detedObjsNum; j++)///allocate each object class of unallocated (id==-3)
        rectIDtmp[j] = -3;

    int closestTab[TRACKER_NUM] = {0}, absxTab[TRACKER_NUM] = {0}, absyTab[TRACKER_NUM] = {0};///table for closestID matching

    for(int i=0; i<TRACKER_NUM; i++){

        if(Tracker[i].trackID>-1)///ignore inactive tracker id=0
            continue;
        // cout<<"///beforeID["<<i<<"] "<<Tracker[i].trackID<<" T0C["<<i<<"]"<<Tracker[i].detect0Count<<endl;
        // Tracker[i].trackID = 0; //reset the predict rectIDtmp as inactive tracker id=0
        Point centerDest(Tracker[i].rectTarget.x+Tracker[i].rectTarget.width/2, Tracker[i].rectTarget.y+Tracker[i].rectTarget.height/2);
        int absx=0, absy=0, initAbs=0; int closestNum=-1;
        //find the most near objects for each tracker
        // cout<<" Target["<<i<<"] "<< Tracker[i].trackID<<" x "<<Tracker[i].rectTarget.x+Tracker[i].rectTarget.width/2<<" y "<<Tracker[i].rectTarget.y+Tracker[i].rectTarget.height/2<<endl;
        for(int j=0; j<detedObjsNum; j++){
            Point centerCand(detedObjs[j].x+detedObjs[j].width/2, detedObjs[j].y+detedObjs[j].height/2);
            // cout<<j<<" absx "<<abs(centerCand.x-centerDest.x)<<" margin "<<(detedObjs[j].width*detedObjs[j].width/3000)<<endl;
            if( ( abs(centerCand.x-centerDest.x) < (detedObjs[j].width*0.5) ) && ( abs(centerCand.y-centerDest.y) < (detedObjs[j].height*0.3) ) ){//find the tracking object
                // cout<<"x "<<centerCand.x<<" margin "<< std::max(10, int(0.25*centerCand.x-55))<<endl;
                if(initAbs==0){
                    // cout<<"chosen "<<j<<" absx "<<abs(centerCand.x-centerDest.x)<<" absy "<<abs(centerCand.y-centerDest.y)<<endl;
                    initAbs++;
                    closestNum = j;//record the closetID
                    absx = abs(centerCand.x-centerDest.x);
                    absy = abs(centerCand.y-centerDest.y);
                }else if( abs(centerCand.x-centerDest.x)+abs(centerCand.y-centerDest.y) < absy+absx ){
                    // cout<<"chosen "<<j<<" absx "<<abs(centerCand.x-centerDest.x)<<" absy "<<abs(centerCand.y-centerDest.y)<<endl;
                    if( absx<(detedObjs[j].width*0.4) && absy<(detedObjs[j].height*0.4) )//set less near objects but for the same tracker class inactive id =0
                        rectIDtmp[closestNum] = 0;
                    closestNum = j;//record the closetID
                    absx = abs(centerCand.x-centerDest.x);
                    absy = abs(centerCand.y-centerDest.y);
                }else{
                    // cout<<"giveup "<<j<<" absx "<<abs(centerCand.x-centerDest.x)<<" absy "<<abs(centerCand.y-centerDest.y)<<endl;
                    if(absx<30 && absy<30)//set less near objects but for the same tracker class inactive id =0
                        rectIDtmp[j] = 0;
                }
            }       
        }



        if(closestNum>=0){
            rectIDtmp[closestNum] = 0; //mark the closest detect object 
            Tracker[i].rectTarget = detedObjs[closestNum];
            // 
            if( Tracker[i].detect0Count == -1)
                Tracker[i].trackID ++;
            else
                Tracker[i].trackID = -1; //active tracker id=-1
        // cout<<" Target["<<i<<"] "<< Tracker[i].trackID<<" x "<<Tracker[i].rectTarget.x+Tracker[i].rectTarget.width/2<<" y "<<Tracker[i].rectTarget.y+Tracker[i].rectTarget.height/2<<endl;
        // cout<<"closestID "<<closestNum<<" rectIDtmp "<<rectIDtmp[closestNum]<< " trackID["<<i<<"] "<< Tracker[i].trackID<<endl;
        }else{
            Tracker[i].trackID--;
        }
        // cout<< "T"<<i<<" "<< Tracker[i].trackID<<" C "<<Tracker[i].detect0Count<<endl;

        //cout<<"\\\\\\afterID["<<i<<"] "<<Tracker[i].trackID<<" T0C["<<i<<"] "<<Tracker[i].detect0Count<<endl;
        // cout<<"Tracker[i].trackID "<<Tracker[i].trackID<<endl;
        if(Tracker[i].detect0Count == -1){// for a new tracker
            if( Tracker[i].trackID < -3*DET0_COUNT_MAX){//too many 0 detection indicate the new tracker is a mistake detection
                Tracker[i].trackID = 0;
                // cout<<"  free tracker_"<<i<<endl;
                Tracker[i].detect0Count=0;

            }
            if(Tracker[i].trackID == -DET0_COUNT_MAX){//the new tracker accumulate enough  detection to certain a target
                Tracker[i].trackID = -1;
                Tracker[i].detect0Count=0;
            }
        }else{
            //if no deteciton of one target prediction for more than 'DET0_COUNT_MAX' times, drop that object
            if( Tracker[i].trackID == -DET0_COUNT_MAX){ //too many 0 detection indicate the target is lost
                    Tracker[i].trackID = 0;
                    // #ifdef SHOWINFO
                    //     cout<<"  free tracker_"<<i<<endl;
                    // #endif
                    Tracker[i].detect0Count=0;
            }
        }

        // cout<<" Target["<<i<<"] "<< Tracker[i].trackID<<endl;
        // waitKey(0);
        //remove the trackTarget that share the same closetID
        if(closestNum>=0){
            if(closestTab[closestNum] == 0){
                closestTab[closestNum] = i+1;
                absxTab[closestNum] = absx;
                absyTab[closestNum] = absy;
            }else if(absx<absxTab[closestNum]&&absy<absyTab[closestNum]){
                Tracker[ closestTab[closestNum]-1 ].trackID = 0;//remove if there is previous traceTarget for the same closestNum
                // #ifdef SHOWINFO
                //     cout<<" rm repeat Tracker "<<closestTab[closestNum]-1<<" ID "<<closestNum<<endl;
                // #endif
                closestTab[closestNum] = i+1;
                absxTab[closestNum] = absx;
                absyTab[closestNum] = absy;
            }else{
                Tracker[i].trackID = 0; //Tracker[i] track the same target, remove it
                // #ifdef SHOWINFO
                //     cout<<" rm repeat Tracker "<<i<<" ID "<<closestNum<<endl;
                // #endif
            }
        }

    }
    
    // for(int i=0; i<detedObjsNum; i++){
    //     cout<<"obj.x "<<detedObjs[i].x+detedObjs[i].width/2<<endl;
    //     cout<<"rectIDtmp[i] "<<rectIDtmp[i]<<endl;
    // }

    int k = 0;
    for(int i=0; i<detedObjsNum; i++){//there is new detection target
        if(rectIDtmp[i]>-3) ///ignore detected object allocated already
            continue;
        for(;k<TRACKER_NUM;k++){
            //cout<<"k"<<k<<endl;
            //cout<< "Tracker["<<k<<"].trackID "<< Tracker[k].trackID<<endl;
            if( Tracker[k].trackID == 0 ){//find the idle tracker to follow the new target
                #ifdef SHOWINFO
                    // cout<<" allocate tracker_"<<k<<endl;
                #endif
                rectIDtmp[i] = 0;
                Tracker[k].rectTarget = detedObjs[i];
                Tracker[k].SetNewTrack (Tracker[k].rectTarget);
                #ifdef TRACK_TRACKER
                    Tracker[k+TRACKER_NUM2-TRACKER_NUM].SetNewTrack (Tracker[k].rectTarget);
                #endif
                //Tracker[k].rectPredict = Tracker[k].rectTarget;
                //cout<<"Ta"<<i<<" "<<setw(2)<<Tracker[i].trackID<<" C "<<Tracker[i].detect0Count<<" x "<<Tracker[i].rectTarget.x<<" w "<<Tracker[i].rectTarget.width<<endl; 
                Tracker[k].trackID = -DET0_COUNT_MAX-NEW_DET_IGNORE;//set a new tracker more 0 dection intially, spare for mistake detection
                Tracker[k].detect0Count = -1; //-1 indicate a new tracker
                if( Tracker[k].trackID == -DET0_COUNT_MAX ){
                    Tracker[k].trackID = -1;
                    Tracker[k].detect0Count=0;
                }
                // cout<<" Target["<<k<<"] "<< Tracker[k].trackID<<" x "<<Tracker[k].rectTarget.x+Tracker[k].rectTarget.width/2<<" y "<<Tracker[k].rectTarget.y+Tracker[k].rectTarget.height/2<<endl;
                // waitKey(0);
                break;
            }
        }
    }
    
    // for (int i = 0; i < TRACKER_NUM; ++i){
    //     cout<< "T"<<i<<" "<< Tracker[i].trackID<<" C "<<Tracker[i].detect0Count<<" x "<<Tracker[i].rectTarget.x+Tracker[i].rectTarget.width/2<<" w "<<Tracker[i].rectTarget.width<<endl;
    // }

    // for (int i = 0; i < TRACKER_NUM; ++i){
    //     if(Tracker[i].trackID<0)
    //         cout<<" Target["<<i<<"] "<< Tracker[i].trackID<<endl;
    // }


    // for (int l = 0; l < TRACKER_NUM; ++l){
    // cout<< "T"<<l<<" "<< Tracker[l].trackID<<" C "<<Tracker[l].detect0Count<<" x "<<Tracker[l].rectTarget.x<<" w "<<Tracker[l].rectTarget.width<<endl;
    // }
    /// do follow 5 prediction to general enouth frame for buffer
    #ifndef TRACK_TRACKER
        for (int i = 0; i < TRACKER_NUM; ++i){
            if(Tracker[i].trackID<0){
                for (int j = 1; j < PREDICT_NUM; ++j){
                    //cout<<endl<<">>Target["<<i<<"] "<<"predit_"<<j<<"<<"<<endl;
                    if(j==1)
                        rPredict[j][i] = Tracker[i].TrackRect(Tracker[i].rectTarget, -1, -1);
                    else
                        rPredict[j][i] = Tracker[i].TrackRect(rPredict[j-1][i], j, 5);
                    statePt[j][i] = Point(Tracker[i].kalman.statePost.ptr<float>(0)[0], Tracker[i].kalman.statePost.ptr<float>(1)[0]);
                }

            }
        }
    #else
        printf("define track tracker\n");
        for (int i = 0; i < TRACKER_NUM; ++i){
            if(Tracker[i].trackID<0){
                for (int j = 1; j < PREDICT_NUM; ++j){
                    //cout<<endl<<">>Target["<<i<<"] "<<"predit_"<<j<<"<<"<<endl;
                    if(j==1)
                        rPredict[j][i] = Tracker[i].TrackRect(Tracker[i].rectTarget, -1, -1);
                    if(j==2||j==3)
                        rPredict[j][i] = Tracker[i].TrackRect(rPredict[j-1][i], j, 5);
                    if(j==4||j==5){
                        Tracker[i+TRACKER_NUM2-TRACKER_NUM].rectPredict = rPredict[j-2][i];
                        rPredict[j][i] = Tracker[i+TRACKER_NUM2-TRACKER_NUM].TrackRect(rPredict[j-2][i], 3, -1);
                    }
                    statePt[j][i] = Point(Tracker[i].kalman.statePost.ptr<float>(0)[0], Tracker[i].kalman.statePost.ptr<float>(1)[0]);
                }

            }
        }
    #endif
    

}

vehDetect::vehDetect(int t)
{
    printf("consturct thread %d\n", t);
    threadTag = t;
}

void vehDetect::InitParam(){
    long int timetag1;
    long int timetag2;
    timetag1 = GetWallTime();
    timeTagPre = GetWallTime();
    cout<<"threadTag "<<threadTag<<endl;
    #if DEBUG
        if(classifierName.length()>2){
            cvVehDet.LoadClassifier(classifierName);
            cout<<"length "<<classifierName.length()<<" use classifier "<<classifierName<<endl;
            // getchar();
        }
        else{
            // cvVehDet.LoadClassifier("cascade_vehicle.xml");
            // cout<<"use default classifier "<<endl;
            cout<<"please input classifier name!"<<endl;
            exit(0);
            // getchar();
        }
    #else

        cvVehDet.LoadClassifier("cascade_vehicle.xml");
    #endif
    //initial trackers
    for(int i=0; i<TRACKER_NUM2; i++){
        VehTrack.Tracker[i].InitKalman();
        // memset(&VehTrack.Tracker,0,sizeof(VehTrack.Tracker));
        // memset(&VehTrack.rPredict,0,sizeof(rPredict));
        VehTrack.Tracker[i].trackID = 0;
        VehTrack.Tracker[i].detect0Count = 0;
        
    }
    //initial npre
    VehTrack.nPre = 0;
    bkpDone = false;
    //std::cout<<"Tracker.trackmark "<<Tracker.trackmark<<std::endl;
    timetag2 = GetWallTime();
    printf("version %s\nClassifier loaded, time: %ldms\n", version, timetag2-timetag1);

}

///process combined thread result
DetResultInfo* vehDetect::ProcessThreadResult(DetResultInfo* threadresult, int threadNum)
{
    #ifdef SHOWINFO2
        static unsigned int timeCount2 = 0;
        static unsigned int timeSum2 = 0;
    #endif

    #ifdef SHOWINFO
        static unsigned int timeCount3 = 0;
        static unsigned int timeSum3 = 0;
    #endif

    T10
    ROI threadRsltVec;
    std::vector<cv::Rect> rectThdRslt;//save Processed Thread Result
    memset(&predictPosResult, 0, sizeof(predictPosResult)); //initialize return result
    for(int i=0; i<PREDICT_NUM; i++){
        predictPosResult[i].type = (ArithmeticType)threadTag;
        predictPosResult[i].alarm.time = 0xFF;
    }


    for (int i=0; i<threadNum; ++i){
        //cout<<"threadresult num "<<threadresult[j].position.RectNum<<endl;
        for(int j=0; j<threadresult[i].position.RectNum; j++){
            //cout<<"threadrect"<<j<<" x "<<threadresult[j].position.Rect[j].Top.x<<" w "<<threadresult[j].position.Rect[j].Width<<endl;
            cv::Rect rectTmp;
            rectTmp.x = threadresult[i].position.Rect[j].Top.x;
            rectTmp.y = threadresult[i].position.Rect[j].Top.y;
            rectTmp.width = threadresult[i].position.Rect[j].Width;
            rectTmp.height = threadresult[i].position.Rect[j].Hight;
            rectThdRslt.push_back(rectTmp);  
        }
    }
     
    /// supress the overlapping rectangular
    #if DEBUG

        if(doNMS)
    #endif
    {
        int minNeighbors = 0;
        double GROUP_EPS = 0.3;
        #if DEBUG
            if(doMutiScaleDet) minNeighbors = 1;
        #endif
        cvVehDet.GroupRectangles( rectThdRslt, minNeighbors, GROUP_EPS );
    }

    int timePeriod = GetWallTime() - timeTagPre;
    timeTagPre = GetWallTime();
    //cout<<"timePeriod "<<timePeriod<<endl;
    int n = timePeriod/FRAME_INTERVAL; ///ignore the predict frame occur during processing 

    if(n>=PREDICT_NUM-1)
        n = PREDICT_NUM-2;
    #if DEBUG
        if(n<1||doPause) n=2;
        skipRsNum = n;
    #else
    if(n<1) n=2;
    #endif

    #ifdef SHOWINFO2
         //for synchronizaton with debug main function
        if(timeCount2%50 == 0)
            cout<<"skipframes = "<<n<<endl;
        ///load start position of current frame base on the the last predict frame that was used 
        // cout<<"pre skipframes = "<<VehTrack.nPre<<endl;
    #endif

    for (int j = 0; j < TRACKER_NUM; ++j){
        VehTrack.Tracker[j].rectTarget = VehTrack.rPredict[VehTrack.nPre][j];
        VehTrack.Tracker[j].kalman.statePost.ptr<float>(0)[0] = VehTrack.statePt[VehTrack.nPre][j].x;
        VehTrack.Tracker[j].kalman.statePost.ptr<float>(0)[1] = VehTrack.statePt[VehTrack.nPre][j].y;
        // if(VehTrack.Tracker[j].trackID<0)
        //cout<<" target["<<j<<"].x "<<VehTrack.Tracker[j].rectTarget.x+VehTrack.Tracker[j].rectTarget.width/2<<" spx "<<j<<" "<<VehTrack.Tracker[j].kalman.statePost.ptr<float>(0)[0]<<endl;
    }
    VehTrack.nPre = n;//update previous n  

    threadRsltVec.PushRect(predictPosResult[0], rectThdRslt, 0);///save detected obj rectangular to predictPosResult[0]

    VehTrack.AllocTrack(rectThdRslt); ///allocate each tracker the trackobject, process multitrack

    n++;

    for(int i=n; i<PREDICT_NUM; i++){
        // cout<<"predictPosResult "<<i<<endl;
        ///save tracked obj rectangular to return result
        #if DEBUG
            for (int j = 0; j < TRACKER_NUM; ++j){
        #else
            for (int j = 0; j < TRACKER_NUM && j < 7; ++j){///return no more than 7 result
        #endif
            if(VehTrack.Tracker[j].trackID<0&&VehTrack.Tracker[j].trackID>-DET0_COUNT_MAX){
                threadRsltVec.PushRect(predictPosResult[i-n+1], VehTrack.rPredict[i][j], -1);//put current predict rect into return result
            }
        }
    }
    
    for (int i = n-1; i > 0; --i){
        predictPosResult[PREDICT_NUM-i] = predictPosResult[PREDICT_NUM-n];
        // for (int j = 0; j < TRACKER_NUM; ++j){
        //     if(VehTrack.Tracker[j].trackID<0){
        //         cout<<"RS["<<PREDICT_NUM-i<<"] RS["<<PREDICT_NUM-n-1<<"]["<<j<<"]"<<" x "<<predictPosResult[PREDICT_NUM-i].position.Rect[j].Top.x+predictPosResult[PREDICT_NUM-i].position.Rect[j].Width/2<<endl;     
        //     }
        // }
        //cout<<"predictPosResult["<<PREDICT_NUM-i<<"] << ["<<PREDICT_NUM-1<<"]"<<endl;
    }
    // printf("mark1\n");
    // cv::Mat frameYUVbkp(720*3/2, 1280, CV_8UC1);

    // for(int i=0; i<predictPosResult[1].position.RectNum; i++){
    //     Rect objRect(predictPosResult[1].position.Rect[i].Top.x, predictPosResult[1].position.Rect[i].Top.y,
    //                  predictPosResult[1].position.Rect[i].Width, predictPosResult[1].position.Rect[i].Hight);
    //     // imshow("Original Image", testImg);
    //     CalcHistYUV(frameYUVbkp, objRect);
    // }
    // cv::Mat frameRGB;
    // frameYUVbkp.data = bufYuv;
    // cv::cvtColor(frameYUVbkp, frameRGB, CV_YUV2BGR_I420);
    // imshow("rgb_back", frameRGB);
    // bkpDone = false;

    /// demarcation mode, show point and its calculate distance from demarcation formula
    #ifdef DEMA
        memset(&predictPosResult, 0, sizeof(predictPosResult)); //initialize return result
        for(int i=0; i<PREDICT_NUM; i++){
            predictPosResult[i].type = (ArithmeticType)threadTag;
            predictPosResult[i].alarm.time = 0xFF;
        }

        for (int i = 0; i < PREDICT_NUM; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                cv::Rect rectDist(640, 720-j*50+10, 2, 2);
                threadRsltVec.PushRect(predictPosResult[i], rectDist, -1);
            }
        }
    #endif

        

    //calculate distance
    double theta=0;
    for(int j = 0; j < PREDICT_NUM; j++){
        for (int i = 0; i < predictPosResult[j].position.RectNum; ++i)
        {
            Point bottomMid(predictPosResult[j].position.Rect[i].Width/2+predictPosResult[j].position.Rect[i].Top.x, 
                predictPosResult[j].position.Rect[i].Hight+predictPosResult[j].position.Rect[i].Top.y);
            // x=641.250000
            // y=325.250000
            
            // theta = -0.0379*bottomMid.y+100.4;
            if(bottomMid.y>0&&bottomMid.y<=330)
                theta=-0.0379*bottomMid.y+100.4; //(0,330)   (35,44) 误差在1m内
            if(bottomMid.y>330&&bottomMid.y<=360)
                theta=-0.0379*bottomMid.y+100.3; //(330,360) (21,35) 误差在1m内
            if(bottomMid.y>360&&bottomMid.y<=400)
                theta=-0.0379*bottomMid.y+100.2; //(360,400) (14,21)误差在1m内
            if(bottomMid.y>400&&bottomMid.y<=720)
                theta=-0.0378*bottomMid.y+100.2; //(400,720) (0,14) 误差0.5m内
            
            double d = 140*tan(theta*3.1415926/180);
            double l = (bottomMid.x-640)*(0.0006*bottomMid.y*bottomMid.y-0.4605*bottomMid.y+86.753); //%(0,360)
            // %l=(x-640)*(0.00006*y^2-0.0628*y+16.03) %(360,720)
            double D = (sqrt(d*d+l*l)-160)/100;
            // cout<<"d "<<d<<" l "<<l<<" theta "<<theta<<" D "<<D<<endl;

            predictPosResult[j].position.Rect[i].distance = D;
        }
    }


    ///calculate speed
    // int speed[predictPosResult[0].position.RectNum];
    // for (int i = 0; i < predictPosResult[0].position.RectNum; ++i){
    //     speed[i] = (predictPosResult[2].position.Rect[i].distance - predictPosResult[1].position.Rect[i].distance) * 1000 / 40;
    // }

    // for(int j = 0; j < PREDICT_NUM; j++){
    //     for (int i = 0; i < predictPosResult[j].position.RectNum; ++i)
    //         predictPosResult[j].position.Rect[i].distance = speed[i];
    // }
    T11
    #ifdef SHOWINFO2
        if(timeCount2%50 == 0){
            cout<<"--------------------------------------------------------"<<endl;
            std::cout<<"processing time12: "<<timeTag11-timeTag10<<std::endl;
            cout<<"period time: "<<timePeriod<<endl;
        }
        timeCount2++;
    #endif
    #ifdef SHOWINFO
        if(timeCount3%50 == 0)
        cout<<"--------------------------------------------------------"<<endl;
        timeCount3++;
    #endif
    

    return predictPosResult;
}


void vehDetect::StopDet(){}


///vehicle detect interface
DetResultInfo vehDetect::ArithmeticInterFace(unsigned char* pYuvBuf, int height, int width, RecCarInfo* inPtr)
{
    #ifdef SHOWINFO2
        static unsigned int timeCount=0;
    #endif
    
    #ifdef SHOWINFO
        static unsigned int timeCount1=0;
        static unsigned int timeCount2=0;
        static unsigned int timeCount3=0;
        static unsigned int timeSum1=0;
        static unsigned int timeSum2=0;
        static unsigned int timeSum3=0;
    #endif

    long int timetag1;
    long int timetag2;
    timeTag3 = 0;
    timeTag6 = 0;
    timeTag9 = 0; 
    timeTag12 = 0; 
    timetag1 = GetWallTime();

   
    cv::Mat frameYUV(height, width, CV_8UC1);
    frameYUV.data = pYuvBuf;
    
    //backup full image for processing
    // bkpDone = true;
    // if(!bkpDone){
    //     pthread_mutex_lock(&mutex);
    //     cv::Mat yuvcolor(720*3/2, 1280, CV_8UC1);
    //     cv::Mat frameRGB;
    //     yuvcolor.data = pYuvBuf;
    //     frameYUVbkp = yuvcolor.clone();
    //     bkpDone = true;
    //     pthread_mutex_unlock(&mutex);
    // }

    // memcpy(frameImg.imageData, pYuvBuf, width*height*sizeof(unsigned char));
    cv::Mat roiImage;
    //int rectCount = 0;
    int roiTop = 0; 
    int roiBtm = 0;
    int roiLeft = 0;
    int roiRight = 0; 
    double scaleRatio = 1;
    bool doRoiDet = true;

    ROI detRoi; 
    detRoi.winNum = 0;
    //set return result information
    DetResultInfo returnRS;
    memset(&returnRS,0,sizeof(returnRS));
    returnRS.type = (ArithmeticType)threadTag;
    returnRS.alarm.time = 0xFF;
    returnRS.position.RectNum = 0;

    if(!cvVehDet.classifierLoadSuccess){
        printf("Cannot detect without classifier!\n");
        return returnRS;
    }
    ///section original frame
    detRoi.results.clear();
    ///to throuth each roi
    for (int i = 0; i < ROINUM_MAX; i++)
    {
        // T1
        // cout<<"threadTag "<<threadTag<<endl;

        ///allocate roi for each thread #modify
        #if 1
            if( ( threadTag==1 && (i==3||i==5) ) 
                || ( threadTag==2 && (i==6||i==0) ) 
                || ( threadTag==3 && (i==7) ) ){
            }else
                continue;
        #else
            if(    ( threadTag==1 && (i==3||i==7) ) 
                || ( threadTag==2 && (i==2||i==5||i==8) ) 
                || ( threadTag==3 && (i==0||i==4||i==6) ) ){
            }else
                continue;
        #endif

        detRoi.winNum = i;
        doRoiDet = true;
        //each case for a roi
        ///set each roi size and scale #modify
        switch(i){
            // case 9: roiTop = 260; roiBtm = 340; roiLeft = 360; roiRight = 730; scaleRatio = 2.1; break;//truck
            case 0: roiTop = 60; roiBtm = 550; roiLeft = 20; roiRight = 1260; scaleRatio = 0.14; break;//truck
            // case 9: roiTop = 40; roiBtm = 600; roiLeft = 20; roiRight = 1140; scaleRatio = 0.09; break;
            case 8: roiTop = 230; roiBtm = 375; roiLeft = 250; roiRight = 880; scaleRatio = 0.54; break;
            case 7: roiTop = 210; roiBtm = 450; roiLeft = 180; roiRight = 990; scaleRatio = 0.38; break;
            case 6: roiTop = 170; roiBtm = 500; roiLeft = 150; roiRight = 1060; scaleRatio = 0.26; break;
            case 5: roiTop = 170; roiBtm = 510; roiLeft = 100; roiRight = 1110; scaleRatio = 0.22; break;
            case 4: roiTop = 150; roiBtm = 540; roiLeft = 70; roiRight = 1110; scaleRatio = 0.20; break;
            case 3: roiTop = 140; roiBtm = 570; roiLeft = 20; roiRight = 1260; scaleRatio = 0.18; break;
            case 2: roiTop = 100; roiBtm = 650; roiLeft = 20; roiRight = 1260; scaleRatio = 0.14; break;//make thread 100ms
            default: doRoiDet = false; break;     
        }
        #ifdef DRAWROI
        if(doRoiDet)
            detRoi.PushRect(returnRS, cv::Rect(roiLeft, roiTop, roiRight-roiLeft, roiBtm-roiTop), scaleRatio);
        #endif       

        #if DEBUG
            if( (i==roiNum || roiNum<0) && doRoiDet )
        #else
            int topOffset = 0;
            int btmOffset = 0;
            int leftOffset = 0;
            int rightOffset = 0;
            double scaleOffset = 0;
            if(doRoiDet) 
        #endif
            {
                //adjust window according to manually hotkey offset
                detRoi.stepSize = 4;

                roiLeft+=leftOffset;
                roiRight+=rightOffset;
                roiTop+=topOffset; 
                roiBtm+=btmOffset;

                if(scaleRatio+scaleOffset>0.1) 
                    scaleRatio+=scaleOffset;

                if(roiLeft<0)
                    leftOffset = 0;
                if(roiRight>frameYUV.cols) 
                    roiRight = frameYUV.cols;
                if(roiTop<0) 
                    roiTop = 0;
                if(roiBtm>frameYUV.rows) 
                    roiBtm = frameYUV.rows;

                ///windows position offset, when sidebar is 30    
                detRoi.winPosLeft = roiLeft + 47; detRoi.winPosTop = roiTop + 25; 
                
                roiImage = detRoi.ExtractROI(frameYUV, roiLeft, roiTop, roiRight, roiBtm, scaleRatio);///get the roi section of trame
                
                vector<Rect> detedObjs = cvVehDet.DetectVehicle(roiImage, scaleRatio);//opencv single-scale process
                //T7   
                detRoi.GetDetResult(detedObjs);///process the roi frame section   
                //T8        
                //timeTag3 += timeTag2-timeTag1;    
                //timeTag6 += timeTag5-timeTag4;  
                //timeTag9 += timeTag8-timeTag7;
                //cout<<"threadTag "<<threadTag<<" i "<<i<<" Num "<<detRoi.results.size()<<endl;
            }
        // T2
        // std::cout<<"roi "<<i<<" processing time: "<<timeTag2-timeTag1<<"\n"<<std::endl;
    }  
    // for(int i=0; i<detRoi.results.size(); i++)
    //     cout<<"i"<<i<<" w "<<detRoi.results[i].width<<endl;
    //T10
    // cout<<"detRoi.results.size() "<<detRoi.results.size()<<endl;
    detRoi.PushRect(returnRS, detRoi.results, 0);///save detectnms rects
    //T11
    // for(int i=0; i<returnRS.position.RectNum; i++)
    // cout<<"i"<<i<<" w "<<returnRS.position.Rect[i].Width<<endl;

    //timeTag12 = timeTag11-timeTag10;
    timetag2 = GetWallTime();
    #ifndef DEMO
        // if(roiNum>=0) cout<<"roiNum"<<roiNum<<endl;
        //std::cout<<"processing time3: "<<timeTag3<<std::endl; 
        //std::cout<<"processing time6: "<<timeTag6<<std::endl;
        //std::cout<<"processing time9: "<<timeTag9<<std::endl;
        //std::cout<<"processing time12: "<<timeTag11-timeTag10<<std::endl;
    #endif
    //std::cout<<"detected objects number: "<<returnRS.position.RectNum<<std::endl;

    #ifdef SHOWINFO
        ///calculate sum time of each thread
        switch(threadTag){
            case 1: timeSum1+=(timetag2-timetag1);
                    timeCount1++;
                    if(timeCount1%50 == 0){
                        std::cout<<threadTag<<"_processing time: "<<timeSum1<<std::endl;
                        timeSum1 = 0;
                    }
                    break;
            case 2: timeSum2+=(timetag2-timetag1); 
                    timeCount2++;
                    if(timeCount2%50 == 0){
                        std::cout<<threadTag<<"_processing time: "<<timeSum2<<std::endl;
                        timeSum2 = 0;
                    }
                    break;        
            case 3: timeSum3+=(timetag2-timetag1); 
                    timeCount3++;
                    if(timeCount3%50 == 0){
                        std::cout<<threadTag<<"_processing time: "<<timeSum3<<std::endl;
                        timeSum3 = 0;
                    }
                    break;
            default: break;
        }
    #endif

    #ifdef SHOWINFO2
        if(timeCount%50 == 0)
            std::cout<<threadTag<<"_processing time: "<<timetag2-timetag1<<std::endl;
        timeCount++;
    #endif

    #ifdef SAVETIMEDATA
        std::ofstream tout("processint_time.txt", ios::app);
        tout<<timetag2-timetag1<<std::endl;
        tout.close();
    #endif

    return returnRS;

}

