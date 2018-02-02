#include <dirent.h>
#include <unistd.h>
//for multi-thread
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <unistd.h>//sleep function

#include "vehDet.hpp"
#include "debug.hpp"

#define STOP_FRAME_NUM -1

vehDetect vehDet1, vehDet2, vehDet3, vehDet4;
DetResultInfo returnRS1, returnRS2, returnRS3, returnRS4;
unsigned char *threadImg1, *threadImg2, *threadImg3, *threadImg4;
int rowThdImg, colThdImg;
sem_t syncMes1, syncMes2, syncMes3, syncMes4;
sem_t syncEnd1, syncEnd2, syncEnd3, syncEnd4;

/// build thread
void *threadDet1(void *arg){
    ///initial detector
    vehDet1.threadTag = 1;
    vehDet1.InitParam();
    while(1){
        sem_wait(&syncMes1);
        //printf("thread1 detect begin\n"); 
        returnRS1 = vehDet1.ArithmeticInterFace(threadImg1, rowThdImg, colThdImg);
        //printf("thread1 detect end\n");
        sem_post(&syncEnd1);
    }
    exit(0);
    pthread_exit(NULL);
}
void *threadDet2(void *arg){
    vehDet2.threadTag = 2;
    vehDet2.InitParam();
    while(1){
        sem_wait(&syncMes2);
        //printf("thread2 detect begin\n");    
        returnRS2 = vehDet2.ArithmeticInterFace(threadImg2, rowThdImg, colThdImg);
        //printf("thread2 detect end\n");
        sem_post(&syncEnd2);
    }
    exit(0);
}
void *threadDet3(void *arg){
    vehDet3.threadTag = 3;
    vehDet3.InitParam();
    while(1){
        sem_wait(&syncMes3);
        //printf("thread3 detect begin\n");    
        returnRS3 = vehDet3.ArithmeticInterFace(threadImg3, rowThdImg, colThdImg);
        //printf("thread3 detect end\n");
        sem_post(&syncEnd3);
    }
    exit(0);
    pthread_exit(NULL);
}
void *threadDet4(void *arg){
    vehDet4.threadTag = 4;
    vehDet4.InitParam();
    while(1){
        sem_wait(&syncMes4);
        //printf("thread4 detect begin\n");    
        returnRS4 = vehDet4.ArithmeticInterFace(threadImg4, rowThdImg, colThdImg);
        //printf("thread4 detect end\n");
        sem_post(&syncEnd4);
    }
    exit(0);
    pthread_exit(NULL);
}

using namespace std;



///////////////////////////////////
//
//      key function:
//
// o/p: om/off capture frame
// v: show original frame
// w s, d c, z x, e r: adjust margin
// u j h k: adjust roi positon
// y i: change sacel size
// b: process the same frame
// n: normally detect
// sapce: pause/next fame
// t: show track
// f: show true det result and pause
// g: skip processing/do not skip
// 0-9: show different roi
// a: do NMS
// m: auto/manually move window
// q: quit
// l: on/off use mulit-detect
// 
///////////////////////////////////




int main(int argc,char** argv)
{
    int res1, res2, res3, res4;//save sem_t result
    pthread_t thread1,thread2, thread3;//, thread4;

    threadImg1 = (unsigned char*)malloc(720*1280*3/2);
    threadImg2 = (unsigned char*)malloc(720*1280*3/2);
    threadImg3 = (unsigned char*)malloc(720*1280*3/2);
    threadImg4 = (unsigned char*)malloc(720*1280*3/2);

    res1 = sem_init(&syncMes1, 0, 0);
    res2 = sem_init(&syncMes2, 0, 0);
    res3 = sem_init(&syncMes3, 0, 0);
    res4 = sem_init(&syncMes4, 0, 0);

    if (res1 != 0){perror("Sem1 init failed");exit(EXIT_FAILURE);}
    if (res2 != 0){perror("Sem2 init failed");exit(EXIT_FAILURE);}
    if (res3 != 0){perror("Sem3 init failed");exit(EXIT_FAILURE);}
    if (res4 != 0){perror("Sem4 init failed");exit(EXIT_FAILURE);}

    res1 = pthread_create(&thread1, NULL, threadDet1, NULL);
    #ifndef SINGLE_THREAD
    res2 = pthread_create(&thread2, NULL, threadDet2, NULL);
    res3 = pthread_create(&thread3, NULL, threadDet3, NULL);
    // res4 = pthread_create(&thread4, NULL, threadDet4, NULL);
    #endif

    if (res1 != 0){perror("Thread1 create failed");exit(EXIT_FAILURE);}
    if (res2 != 0){perror("Thread2 create failed");exit(EXIT_FAILURE);}
    if (res3 != 0){perror("Thread3 create failed");exit(EXIT_FAILURE);}
    //if (res4 != 0){perror("Thread4 create failed");exit(EXIT_FAILURE);}

    #ifndef DEBUG
        int res = pthread_mutex_init(&mutex, NULL);
        if(res!=0){perror("Mutex init failed!"); exit(EXIT_FAILURE);}
    #endif

#ifdef DEMO
    VideoWriter videoWriter ("vehDetection.avi", CV_FOURCC('X', 'V', 'I', 'D'), 24, Size(1280, 720), true);
#endif
    //doShowRoi = true;
    //prermeters to check it's video or picture
    char* viedoInput = argv[1];
    bool isInitFrame = true;

    if(argc<3||(argv[2][0]!='-')){
        cout<<"identify input type v/i"<<endl;
        exit(0);
    }

    #if DEBUG
        if(argc==4)
            classifierName = std::string(argv[3]);
    #endif

    bool isVideo = (int)argv[2][1] - 105;
    #ifdef PROCESSPIC
    bool isPicture = !isVideo;
    struct dirent *dirt = NULL;
    DIR *dirp = NULL;
    #endif

    //peremeter for sample saving
    bool doSingleFrame = false;
    bool doFrameSample = false;
    bool doDetSample = false;
    //int posImgi = 0;
    //int negImgi = 0;
    char c = 0;
    int p = 5;
    
    cv::Mat frameImg;
    cv::Mat framePre;
    cv::Mat framePre2;  
    cv::Mat imgYUV;
    cv::VideoCapture capture;

    DetResultInfo threadResultMeg[4];
    
    DetResultInfo returnRS[6];
    DetResultInfo* returnRSTmp = returnRS; ///initial  funtion result and result backup


    ///if it's video new capture
    if(isVideo){    
        if(viedoInput) capture.open(argv[1]);
        else std::cout<<"fail to open video"<<std::endl;
        if (!capture.isOpened()){
            printf("Could not open video");
            //printf("cant open camera ! \n");
            exit(-1);
        }
        std::cout<<"load video: "<<argv[1]<<std::endl;
    }
    ///if it's picture
#ifdef PROCESSPIC
    if(isPicture){    
        if ( ( dirp = opendir(argv[1]) ) == NULL ){
            printf ("open dir %s failed\n",argv[1]);
            return 0;
        } 
        if ( chdir(argv[1]) )
            std::cout<<"enter "<<argv[1]<<std::endl;
        else     
            std::cout<<"cannot enter "<<argv[1]<<std::endl; 
    }
#endif
    ///set parameters about detection skip of frames
    int frameskip = 1;
    int frameCount = frameskip;
    int continFrame = 1;
    frameCount--;
    ///set timetag to calculate time consumimg
    long int timetag1;
    long int timetag2;


    //begin detect circulation
    while(continFrame){ 


        timetag1 = GetTimeTag(); 
        ///capture keyboard value, change setting;
        #if DEBUG
        {
            if( frameTag>0 && frameTag==STOP_FRAME_NUM ){
                doPause = true;
                isInitFrame = true;
            }
            if(doPause){
                GetRectInfo(frameImg, framePre);
                c = cv::waitKey(0);
            }else 
                c = cv::waitKey(1);
           
            if (c == 'g') {if(frameskip == 1) frameskip = 1000; else frameskip = 1; frameCount = frameskip;}

            if (c == 'q') {std::cout<<"Quit"<<std::endl; quit = true; exit(0);} 
      
            if (c == 'b') {if(doPause) isInitFrame = false; else isInitFrame = true; doSingleFrame = true; doPause = false; }   
            if (c == ' ') {doPause = true;  isInitFrame = true; if(doSingleFrame) doSingleFrame = false; }
            if (c == 'n') {doPause = false; doSingleFrame = false; isInitFrame = true;}
            if (c == 't') {doShowTrack = !doShowTrack;}
            if (c == 'f') {doShowDet = !doShowDet;} //doShowRoi = !doShowRoi;/show detecting area
            if (c == 'v') {doShowOri = !doShowOri;} ///show original frame
            if (c == 'l') {doMutiScaleDet = !doMutiScaleDet;  if(doPause) isInitFrame = false; else isInitFrame = true; doSingleFrame = true; doPause = false;} //do multicale detect  
            if (c == 'm') {doMoveWin = !doMoveWin;}
            if (c >= '0'&& c<= '9') {roiNum = (int)(c-'0'); doShowRoi = !doShowRoi; p=skipRsNum;} 
            if (c == '`') {doShowRoi = !doShowRoi; roiNum = -1; p=skipRsNum;} // 
            
            if (c == 'w') {doShowSize = true; topOffset-=10;}///+top        
            if (c == 's') {doShowSize = true; topOffset+=10;}///-top                    
            if (c == 'c') {doShowSize = true; btmOffset+=10;}///+bottom              
            if (c == 'd') {doShowSize = true; btmOffset-=10;}///-bottom                      
            if (c == 'z') {doShowSize = true; leftOffset-=10;}///+left             
            if (c == 'x') {doShowSize = true; leftOffset+=10;}///-left                    
            if (c == 'r') {doShowSize = true; rightOffset+=10;}///+right         
            if (c == 'e') {doShowSize = true; rightOffset-=10;}///-right  
            if (c == 'a') {topOffset=0; btmOffset=0; leftOffset=0; rightOffset=0; scaleOffset=0;}

            if (c == 'u') {doShowSize = true; topOffset-=10; btmOffset-=10;}///move up  
            if (c == 'j') {doShowSize = true; topOffset+=10; btmOffset+=10;}///move down
            if (c == 'h') {doShowSize = true; leftOffset-=10; rightOffset-=10;}///move left   
            if (c == 'k') {doShowSize = true; leftOffset+=10; rightOffset+=10;}///move right 
            if (c == 'y') {doShowSize = true; scaleOffset-=0.01;}///move down
            if (c == 'i') {doShowSize = true; scaleOffset+=0.01;}///move down

            if (c == 'o') {//frameskip = 10; frameCount = frameskip; 
            doFrameSample = true;}/// to save negsample  
            if (c == 'p') {//frameskip = 10; frameCount = frameskip; 
            doDetSample = false;}/// to save negsample          
            if(frameCount<=0)
                frameCount = frameskip;
            frameCount--;
        }
        #else
            bool doPause = false;

 
            bool doShowOri = true;

            bool doMoveWin = true;
            int skipRsNum = 0;
             waitKey(1);
        #endif




        #ifdef DEMO

            doShowTrack = true;
        #endif

        if(isVideo){  
            if(!doSingleFrame && (!doPause || isInitFrame) ){
                capture >> frameImg;
                if(frameImg.rows>800)
                    cv::resize(frameImg,frameImg, Size(1280, 720), 0, 0);
                framePre = frameImg;
            }   
            isInitFrame = false;                     
            frameImg = framePre;              
            framePre = frameImg.clone();
        } 
        /*
        if(isVideo){  
            if(isInitFrame){
                capture >> frameImg;
                framePre = frameImg;
            }   
            if(doSingleFrame||doPause){  
                isInitFrame = false;                     
                frameImg = framePre;              
                framePre = frameImg.clone();
                //imshow("pre", framePre);
                //cvWaitKey(0);
            }
        }
        */
        #ifdef  PROCESSPIC        
        if(isPicture){
            if(!doSingleFrame)      
                doPause = true; 
            if(isInitFrame){
                if( ( dirt = readdir(dirp) ) != NULL ) 
                    frameImg = cv::imread(dirt->d_name);
                else 
                    continFrame = 0; 
                framePre = frameImg;        
            }         
            if(doSingleFrame||doPause){
                isInitFrame = false;                 
                frameImg = framePre;
                framePre = frameImg.clone();
            }
            std::cout<<"read: "<<dirt->d_name<<std::endl;                              
            //cout<<"isInitFrame "<<isInitFrame<<" doSingleFrame "<<doSingleFrame<<endl;
        }
        #endif
        if(frameImg.empty()){
            printf("error frame!"); 
            break;
        } 

        if(doDetSample)
            framePre2 = frameImg.clone();

        cv::cvtColor(frameImg, imgYUV, CV_BGR2YUV_I420);
        // cout<<"imgYUV.cols "<<imgYUV.cols<<"imgYUV.rows "<<imgYUV.rows<<endl;
        // imshow("imgYUV", imgYUV);
        // waitKey(0);        
        // cv::cvtColor(frameImg, imgYUV, CV_BGR2GRAY);
        ///process the image  
        if(frameCount == 0){   
            ///process multi thread result
            //cout<<"skipRsNum "<<skipRsNum<<endl;
            if(p<skipRsNum){
                p++;
            }else{
                p=1;
                memset(&returnRS, 0 , sizeof(returnRS));
                for (int j = 0; j < 6; ++j)
                {
                    returnRS[j] = returnRSTmp[j];
                }
                ///use multi-thread to run detect
                rowThdImg = imgYUV.rows;
                colThdImg = imgYUV.cols;

                memcpy(threadImg1, (unsigned char*)imgYUV.data, (imgYUV.rows)*(imgYUV.cols)*(sizeof(unsigned char)));
                memcpy(threadImg2, (unsigned char*)imgYUV.data, (imgYUV.rows)*(imgYUV.cols)*(sizeof(unsigned char)));
                memcpy(threadImg3, (unsigned char*)imgYUV.data, (imgYUV.rows)*(imgYUV.cols)*(sizeof(unsigned char)));
                memcpy(threadImg4, (unsigned char*)imgYUV.data, (imgYUV.rows)*(imgYUV.cols)*(sizeof(unsigned char)));

                sem_post(&syncMes1);
                #ifndef SINGLE_THREAD
                sem_post(&syncMes2);
                sem_post(&syncMes3);
                //sem_post(&syncMes4);
                #endif
                sem_wait(&syncEnd1);
                #ifndef SINGLE_THREAD
                sem_wait(&syncEnd2);
                sem_wait(&syncEnd3);
                //sem_wait(&syncEnd4);
                #endif
                threadResultMeg[0] = returnRS1;      
                threadResultMeg[1] = returnRS2;
                threadResultMeg[2] = returnRS3;
                // cout<<"threadResultMeg num "<<threadResultMeg[0].position.RectNum<<endl;
                // cout<<"threadResultMeg num "<<threadResultMeg[1].position.RectNum<<endl;
                // cout<<"threadResultMeg num "<<threadResultMeg[2].position.RectNum<<endl;

                memset(returnRSTmp, 0, 6*sizeof(DetResultInfo));
                returnRSTmp = vehDet1.ProcessThreadResult(threadResultMeg, 3);
            } 
            #ifdef SHOWINFO2
                #ifdef SHOWINFO
                cout<<"predictCount "<<p<<endl;        
                cout<<"return Rect Num "<<returnRS[p].position.RectNum<<endl;
                #endif
            #endif
            // DetResultInfo returnRS[p] = vehDet.ArithmeticInterFace((unsigned char*)imgYUV.data, imgYUV.rows, imgYUV.cols);

            // for(int j=0; j<6; j++){
            //     cout<<"actual width "<<j<<" "<<returnRS[j].position.Rect[0].Width<<endl;
            // }

            #if DEBUG
                ///draw current deteted object
                if(p==1&&doShowDet){
                    if(returnRSTmp[0].position.RectNum>0){
                        framePre = frameImg.clone();
                        doPause = true; 
                        isInitFrame = true;
                    }

                    for(int j=0;j<returnRSTmp[0].position.RectNum&&j<RECT_NUM_MAX;j++){///draw retangulars}
                        //cout<<"rectID1["<<j<<"] "<<rectID1[j]<<" w "<<returnRS[p].position.Rect[j].Width<<endl;
                        double left = returnRSTmp[0].position.Rect[j].Top.x ;
                        double top = returnRSTmp[0].position.Rect[j].Top.y;
                        double right = returnRSTmp[0].position.Rect[j].Width + left;
                        double bottom = returnRSTmp[0].position.Rect[j].Hight + top;              

                        cv::Scalar rectClr;
                        rectClr = CV_RGB(0,0,255);

                        // double scanWinWidth = 36, finalWidth = returnRS[0].position.Rect[j].Width;
                        // double scaledRate = scanWinWidth/finalWidth;
                        // if(rectID1[j]!=0)
                        //     {scaledRate = rectID1[j];}
                        char rectLabel[10];
                        sprintf(rectLabel, "%.2f", returnRS[0].position.Rect[j].distance);
                        

                        if(!doShowTrack){
                            cv::putText( frameImg, rectLabel, cv::Point(right,top+16), cv::FONT_HERSHEY_PLAIN, 1, rectClr, 2 );
                            cv::rectangle( frameImg, cvPoint(left,top), cvPoint(right,bottom), rectClr, 4 );
                            cv::circle(frameImg, cv::Point( (left+right)/2, (top+bottom)/2), 3, rectClr, 4); 
                        }
                    }

                }

                ///draw precioud deteted object
                for(int j=0;j<returnRS[0].position.RectNum&&j<RECT_NUM_MAX;j++){///draw retangulars}
                    // cout<<"rectID1["<<j<<"] "<<rectID1[j]<<" x "<<returnRS[0].position.Rect[j].Top.x<<" w "<<returnRS[p].position.Rect[j].Width<<endl;
                    double left = returnRS[0].position.Rect[j].Top.x ;
                    double top = returnRS[0].position.Rect[j].Top.y;
                    double right = returnRS[0].position.Rect[j].Width + left;
                    double bottom = returnRS[0].position.Rect[j].Hight + top;              

                    cv::Scalar rectClr;
                    rectClr = CV_RGB(255,69,0);

                    // double scanWinWidth = 36, finalWidth = returnRS[0].position.Rect[j].Width;
                    // double scaledRate = scanWinWidth/finalWidth;
                    // if(rectID1[j]!=0)
                    //     {scaledRate = rectID1[j];}
                    char rectLabel[10];
                    sprintf(rectLabel, "%.2f", returnRS[0].position.Rect[j].distance);
                    if(!doShowTrack){
                        cv::putText( frameImg, rectLabel, cv::Point(right,top+8), cv::FONT_HERSHEY_PLAIN, 1, rectClr, 2 );
                        cv::rectangle( frameImg, cvPoint(left,top), cvPoint(right,bottom), rectClr, 4 );
                        cv::circle(frameImg, cv::Point( (left+right)/2, (top+bottom)/2), 3, rectClr, 4); 
                    }
                }
            #endif
            //draw predict objects
            // if(returnRS[p].position.RectNum>0)
            //     returnRS[p].position.RectNum=1;
            int pp = p;
            // pp+=2;
            #ifdef TRACK_TRACKER

                pp+=2;
            #endif



            // cout<<"p "<<pp<<endl;
            for(int j=0;j<returnRS[pp].position.RectNum&&j<RECT_NUM_MAX;j++){///draw retangulars
                // cout<<"actual width "<<p<<" "<<returnRS[p].position.Rect[0].Width<<endl;
                double left = returnRS[pp].position.Rect[j].Top.x ;
                double top = returnRS[pp].position.Rect[j].Top.y;
                double right = returnRS[pp].position.Rect[j].Width + left;
                double bottom = returnRS[pp].position.Rect[j].Hight + top;              
                ///save detected obj pictures
                // cout<<"returnRS[p].position.RectNum "<<returnRS[p].position.RectNum<<endl;
                //cout<<"left "<<left<<" top "<<top<<" right "<<right<<" bottom "<<bottom<<endl;
                cv::Scalar rectClr = CV_RGB(0,255,0);       
                // double scanWinWidth = 36, finalWidth = returnRS[p].position.Rect[j].Width;
                // double scaledRate = scanWinWidth/finalWidth;
                // if(rectID1[j]!=0)
                //     {scaledRate = rectID1[j];}
                char rectLabel[10];
                sprintf(rectLabel, "%.2f", returnRS[pp].position.Rect[j].distance);
                // if(!doShowTrack || rectID1[j] < 0)
                {
                    cv::putText( frameImg, rectLabel, cv::Point(right,top), cv::FONT_HERSHEY_PLAIN, 1, rectClr, 2 );
                    cv::rectangle( frameImg, cvPoint(left,top), cvPoint(right,bottom), rectClr, 4 );
                    cv::circle(frameImg, cv::Point( (left+right)/2, (top+bottom)/2), 3, rectClr, 4); 
                }
                
                #if DEBUG
                    if(doDetSample){
                        cv::imshow("show",frameImg);
                        c = cv::waitKey(200);
                        if (c == 'g') {frameskip = 1; frameCount = frameskip;}
                        if (c == 't') {frameskip = 10000; frameCount = frameskip;}
                        if (c == ' ') {cv::waitKey(0);}
                        if (c == 'q') {std::cout<<"Quit"<<std::endl; exit(1);}  
                        char posImgPath[]= "../testimg/posdata";                  
                        SaveSample(posImgPath, frameImg, Rect(left, top, returnRS[p].position.Rect[j].Width, returnRS[0].position.Rect[j].Hight), Size(), true);
                        frameImg = framePre2.clone();
                
                    }
                #endif
            }

        }
                        
        #if DEBUG
            if(doFrameSample){
                char negImgPath[]= "../testimg/negdata";
                SaveSample(negImgPath, frameImg, Rect(), Size());
                doFrameSample = !doFrameSample;
            }
        #endif

        if(doShowOri){
            #ifdef DEMO
                if(frameTag>0){
                    videoWriter.write(frameImg);
                    frameTag--;
                }
                else
                    break;
            #endif
            //cv::namedWindow("show", 0);
            cv::imshow("Detection",frameImg);
            if(doMoveWin)
               cv::moveWindow("Detection", 0, 0);
        }
        else
            cv::destroyWindow("Detection");

        #if DEBUG
            for(int i=0; i<ROINUM_MAX; i++){
                char winName[50] = {0};
                if( ( (!doShowRoi) && (doShowRoiPre) ) || ( (i!=roiNum) && roiNum >= 0 ) ){
                    //cout<<"close window"<<i<<winName<<endl; 
                    sprintf(winName, "detRoi_%d", i);
                    cv::destroyWindow(winName);
                }
            }
        
            doShowRoiPre = doShowRoi;
        #endif

        if(doSingleFrame) 
            printf("doSingleFrame\n"); 

timetag2 = GetTimeTag();


        if( ( !doSingleFrame && (c<'0'||c>'9') && (c<'`'||c >'z') ) || c=='n')
            frameTag++;

        #ifndef DEMO
            #ifdef SHOWINFO2
            #ifdef SHOWINFO
            printf("frame no.%ld\n", frameTag);
            printf("program time: %ldms\n\n\n\n", timetag2-timetag1); 
            #endif
            #endif
        #endif 
        ///printf("skipframe: %d frameCount = %d\n", frameskip, frameCount);
        ///cvReleaseImage(&frameImg);  
    }
    printf("end\n");
    free(threadImg1);       
    free(threadImg2);
    free(threadImg3);
    free(threadImg4);
    return 0;
}