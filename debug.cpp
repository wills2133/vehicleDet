#include <vehicleCV.hpp>
#include "debug.hpp"


long int frameTag = 0;
long int timeTag1 = 0;
long int timeTag2 = 0;
long int timeTag3 = 0;
long int timeTag4 = 0;
long int timeTag5 = 0;
long int timeTag6 = 0;
long int timeTag7 = 0;
long int timeTag8 = 0;
long int timeTag9 = 0;
long int timeTag10 = 0;
long int timeTag11 = 0;
long int timeTag12 = 0;


long int GetTimeTag()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return time.tv_usec/1000 + time.tv_sec*1000;
}

#if DEBUG

    int roiNum = -1;
    bool doShowRoi = !true;
    bool doMutiScaleDet = false;
    bool doPause = false;
    bool doShowTrack = false;
    bool debugMark = false;
    int debugCount = 0;
    int skipRsNum = 0;

    int sampNum = 0;
    int scaledPos = 0;
    int scaledPosStep = 0;

    int topOffset = 0;
    int btmOffset = 0;
    int leftOffset = 0;
    int rightOffset = 0;
    double scaleOffset = 0;
    //bool doPause = false;
    bool quit = false;
    bool doSetRoiSize = false;
    bool doShowDet = false;
    bool doShowOri = true;
    bool doMoveWin = true;
    bool doBlockResize = false;
    bool doShowSize = false;
    //bool doShowRoi = false;
    bool doShowRoiPre = false;
    bool doGetRoiSize = false;
    bool doNMS = true;
    // bool doMutiScaleDet = false;
    // bool doShowTrack = true;
    bool doPredict = true;
    //bool doNewTrac = true;
    std::string classifierName;
    pthread_mutex_t mutex;

    double rectID1[50];
    double rectID2[50];
    double rectID3[50];
    double rectID4[50];
    //int roiNum = -1;

    //bool doMoveWin = true;



    

    #ifdef SAVETIMEDATA
        std::ofstream tout("processint_time.txt");
    #endif



    #define NBINS 9
    #define THETA 180 / NBINS
    #define CELLSIZE 20
    #define BLOCKSIZE 2
    #define R (CELLSIZE * (BLOCKSIZE) * 0.5)

    cv::Mat imgToShow;
    cv::Mat imgToSave; 
    cv::Rect selection=Rect(0,0,0,0); 
    cv::Point startPoint;
    cv::Point endPoint;
    int width = 36;
    int height = 36; 
    double ratioBtn = 1;
    bool selectObject = false;
    bool rightDown = false;
    bool run = false;


    ///left down left up, positive sample
    ///right down right up, negitive sample 36*36
    ///right down left up, negitive sample  raw size
    void MouseEvent(int event, int x, int y, int flags, void* data)
    {
        if(doPause == false)
            return;

        switch( event )
        {
        case CV_EVENT_RBUTTONDOWN:
            selectObject = true;
            rightDown = true;
            startPoint = Point(x,y);
            printf("L%d,T%d,r%.2f\n", startPoint.x, startPoint.y, ratioBtn);
            break;

        case CV_EVENT_RBUTTONUP:
            selectObject = false;
            if( selection.height>30 && selection.height>30 ){
                char posImgPath[] = "../testimg/negdata";  
                SaveSample(posImgPath, imgToSave, Rect(selection.x, selection.y, selection.width, selection.height), Size(36, 36));
            };
            break;

        case CV_EVENT_LBUTTONDOWN:
            selectObject = true;
            if(rightDown && selection.height>30 && selection.height>30){
                char posImgPath[] = "../testimg/negdata";  
                SaveSample(posImgPath, imgToSave, Rect(selection.x, selection.y, selection.width, selection.height), Size(0, 0));
            }
            rightDown = false;
            startPoint = Point(x,y);
            printf("L%d,T%d,r%.2f\n", startPoint.x, startPoint.y, ratioBtn);
            break;

        case CV_EVENT_LBUTTONUP:
            selectObject = false;
            if( selection.height>30 && selection.height>30 ){
                char posImgPath[] = "../testimg/posdata";  
                SaveSample(posImgPath, imgToSave, Rect(selection.x, selection.y, selection.width, selection.height), Size(32, 32));
                char posImgPath2[] = "../testimg/posdata_36x36";  
                SaveSample(posImgPath2, imgToSave, Rect(selection.x, selection.y, selection.width, selection.height), Size(36, 36));
            }
            break;

        // case CV_EVENT_RBUTTONDOWN:
        //     cv::resize(imgToSave, imgToSave, cv::Size(), 0.8, 0.8); 
        //     //width = width * 0.8;
        //     //height = height *0.8;
        //     ratioBtn = ratioBtn * 0.9;
        //     cv::imshow("Detection", imgToSave);
        //     break;

        // case CV_EVENT_MBUTTONDOWN:
        //     cv::resize(imgToSave, imgToSave, cv::Size(), 1.2, 1.2); 
        //     //width = width * 1.2;
        //     //height = height * 1.2;
        //     ratioBtn = ratioBtn * 1.1;
        //     cv::imshow("Detection", imgToSave);
        //     break;

        case CV_EVENT_MBUTTONDOWN:
            startPoint = Point(x,y);
            selectObject = true;
            break;

        case CV_EVENT_MBUTTONUP:
            selectObject = false;
            run = true;
            break; 
        }

        if( selectObject )//只有当鼠标左键按下去时才有效，然后通过if里面代码就可以确定所选择的矩形区域selection了
        {
            selection.x = MIN(x, startPoint.x );//矩形左上角顶点坐标
            selection.y = MIN(y, startPoint.y);
            selection.width = MIN(std::abs(x - startPoint.x), std::abs(y - startPoint.y));//矩形宽
            selection.height = MIN(std::abs(x - startPoint.x), std::abs(y - startPoint.y));//矩形高
            endPoint.x = x;
            endPoint.y = y-2;
            // selection &= Rect(0, 0, imgToShow.cols, imgToShow.rows);//用于确保所选的矩形区域在图片范围内
        }

        Mat imgToShow2 = imgToShow.clone();

        if( selection.width > 5 && selection.height > 5 && selectObject)
        {
            char charText[100];
            char charText2[100];
            sprintf(charText, "L%d,T%d", startPoint.x, startPoint.y);
            sprintf(charText2, "R%d,B%d,W%d,H%d,r%.2f", x, y, selection.width, selection.height, (double)32/selection.width);
            cv::putText(imgToShow2, charText, Point(startPoint.x, startPoint.y-2), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 2);
            cv::putText(imgToShow2, charText2, endPoint, cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 2);
            cv::rectangle(imgToShow2, selection, Scalar(0, 255, 0),2,3,0);
        }

        if(selection.width>10&&run){
            if(selection.width % 2 != 0 )
                selection.width += 1;
            if(selection.height % 2 != 0 )
                selection.height += 1;
            run = false;

            Mat testImg;
            imgToSave(selection).copyTo(testImg);

            if (!testImg.data){
                printf("wrong testImg\n");
                return;
            }
            // imshow("Original Image", testImg);


            ShowHistYUV(testImg);
            // 转换成hsv图像
            // cv::Mat hsv;
            // cvtColor(testImg, hsv, CV_BGR2HSV);
            // imshow("hsv", hsv);
            // 分水岭操作
            // int numOfSegments = 0;
            // Mat segments = watershedSegment(testImg, numOfSegments);
            // // 分割排序减少过分割
            // segMerge(testImg, segments, numOfSegments);
            // // 显示分割图像
            // cv::Mat wshed = displaySegResult(segments, numOfSegments);
            // // 显示分割后合并图像
            // Mat wshedWithImage = displaySegResult(segments, numOfSegments, testImg);
            // imshow("Merged segments", wshed);
            // imshow("Merged segments with testImg", wshedWithImage);

            // std::vector<Mat> HOGFeatureMat = cacHOGFeature(testImg);
            ShowHistogram(testImg);

            waitKey(0);
        }
        

        cv::imshow("Detection", imgToShow2);
    }

    void GetRectInfo(cv::Mat imgDeted, cv::Mat imgOrig)
    {
    	//std::cout<<"getframeinfo"<<std::endl;
        // printf("use mouse to get information\n");
    	imgToShow = imgDeted;
        imgToSave = imgOrig;
    	cv::setMouseCallback("Detection", MouseEvent, 0);
    }

    ////////////////////////////
    //   save sample
    ////////////////////////////

    void SaveSample(char* path, cv::Mat originImg, cv::Rect r, cv::Size sampSize, bool doNameInfo)
    {

        cv::Mat sample(r.height, r.width, originImg.type());
        if(r.width>0){
            originImg(r).copyTo(sample);
        }else
            sample = originImg;

        if(sampSize.width>0){
            cv::resize(sample, sample, sampSize, 0, 0);
        }

        //cout<<"path "<<path<<endl;
        int indxNum;
        std::ostringstream pathIndx, pathFiles;
        
        pathIndx<<path<<"/index.txt";
        ifstream readIndx(pathIndx.str().c_str(), ios::in);
        if(readIndx.peek() == EOF) indxNum=0;
        else readIndx>>indxNum;
        readIndx.close();
        
        if(doNameInfo)
            pathFiles<<path<<"/"<<r.y+r.height<<"_"<<r.x+r.width/2<<"_"<<r.height<<"_"<<r.width<<".bmp";
        else
            pathFiles<<path<<"/("<<indxNum<<").bmp";
        imwrite(pathFiles.str().c_str(), sample);
        std::cout<<pathFiles.str()<<" saved!"<<std::endl;

        indxNum++;
        ofstream writeIndx(pathIndx.str().c_str(), ios::out);
        writeIndx<<indxNum<<endl;
        writeIndx.close();
    }

    ////////////////////////////
    //   graph cut
    ////////////////////////////

    // 分割合并
    void segMerge(Mat & image, Mat & segments, int & numSeg)
    {
        // 对一个分割部分进行像素统计
        vector<Mat> samples;
        // 统计数据更新
        int newNumSeg = numSeg;
        // 初始化分割部分
        for (int i = 0; i <= numSeg; i++)
        {
            Mat sampleImage;
            samples.push_back(sampleImage);
        }
        // 统计每一个部分
        for (int i = 0; i < segments.rows; i++)
        {
            for (int j = 0; j < segments.cols; j++)
            {
                // 检查每个像素的归属
                int index = segments.at<int>(i, j);
                if (index >= 0 && index<numSeg)
                {
                    samples[index].push_back(image(Rect(j,i,1,1)));
                }
            }
        }
        // 创建直方图
        vector<MatND> hist_bases;
        Mat hsv_base;
        // 直方图参数设置
        int h_bins = 35;
        int s_bins = 30;
        int histSize[] = { h_bins, s_bins };
        // hue 变换范围 0 to 256, saturation 变换范围0 to 180
        float h_ranges[] = { 0, 256 };
        float s_ranges[] = { 0, 180 };
        const float* ranges[] = { h_ranges, s_ranges };
        // 使用第0与1通道
        int channels[] = { 0, 1 };
        // 直方图生成
        MatND hist_base;
        for (int c = 1; c < numSeg; c++)
        {
            if (samples[c].dims>0){
                // 将区域部分转换成hsv
                cvtColor(samples[c], hsv_base, CV_BGR2HSV);
                // 直方图统计
                calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
                // 直方图归一化
                normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
                // 添加到统计集
                hist_bases.push_back(hist_base);
            }
            else
            {
                hist_bases.push_back(MatND());
            }
            hist_base.release();
        }
        double similarity = 0;
        vector<bool> mearged;
        for (unsigned int k = 0; k < hist_bases.size(); k++)
        {
            mearged.push_back(false);
        }
        // 统计每一个部分的直方图相似
        for (unsigned int c = 0; c<hist_bases.size(); c++)
        {
            for (unsigned int q = c + 1; q<hist_bases.size(); q++)
            {
                if (!mearged[q])
                {
                    if (hist_bases[c].dims>0 && hist_bases[q].dims>0)
                    {
                        similarity = compareHist(hist_bases[c], hist_bases[q], CV_COMP_BHATTACHARYYA);
                        if (similarity>0.8)
                        {
                            mearged[q] = true;
                            if (q != c)
                            {
                                //区域部分减少
                                newNumSeg--;
                                for (int i = 0; i < segments.rows; i++)
                                {
                                    for (int j = 0; j < segments.cols; j++)
                                    {
                                        int index = segments.at<int>(i, j);
                                        // 合并
                                        if (index == q){
                                            segments.at<int>(i, j) = c;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        numSeg = newNumSeg;
    }

    //floodfill
    Mat watershedSegment(Mat & image, int & noOfSegments)
    {
        Mat gray;
        Mat ret;
        cvtColor(image, gray, CV_BGR2GRAY);
        imshow("Gray Image", gray);
        // 阈值操作
        threshold(gray, ret, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
        imshow("Image after OTSU Thresholding", ret);
        // 形态学开操作
        morphologyEx(ret, ret, MORPH_OPEN, Mat::ones(9, 9, CV_8SC1), Point(4, 4), 2);
        imshow("Thresholded Image after Morphological open", ret);
        // 距离变换
        Mat distTransformed(ret.rows, ret.cols, CV_32FC1);
        distanceTransform(ret, distTransformed, CV_DIST_L2, 3);
        // 归一化
        normalize(distTransformed, distTransformed, 0.0, 1, NORM_MINMAX);
        imshow("Distance Transformation", distTransformed);
        // 阈值化分割图像
        threshold(distTransformed, distTransformed, 0.1, 1, CV_THRESH_BINARY);
        //归一化统计图像到0-255
        normalize(distTransformed, distTransformed, 0.0, 255.0, NORM_MINMAX);
        distTransformed.convertTo(distTransformed, CV_8UC1);
        imshow("Thresholded Distance Transformation", distTransformed);
        //计算标记的分割块
        // int i, j;
        int compCount = 0;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(distTransformed, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        if (contours.empty())
            return Mat();
        Mat markers(distTransformed.size(), CV_32S);
        markers = Scalar::all(0);
        int idx = 0;
        // 绘制区域块
        for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
            drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
        if (compCount == 0)
            return Mat();
        //计算算法的时间复杂度
        double t = (double)getTickCount();
        watershed(image, markers);
        t = (double)getTickCount() - t;
        printf("execution time = %gms\n", t*1000. / getTickFrequency());
        Mat wshed = displaySegResult(markers, compCount);
        imshow("watershed transform", wshed);
        noOfSegments = compCount;
        return markers;
    }

    //show pic in colors
    Mat displaySegResult(Mat  segments, int numOfSegments, Mat  image)
    {
        Mat wshed(segments.size(), CV_8UC3);
        // 创建对于颜色分量
        vector<Vec3b> colorTab;
        for (int i = 0; i < numOfSegments; i++)
        {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        //应用不同颜色对每个部分
        for (int i = 0; i < segments.rows; i++)
        {
            for (int j = 0; j < segments.cols; j++)
            {
                int index = segments.at<int>(i, j);
                if (index == -1)
                    wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
                else if (index <= 0 || index > numOfSegments)
                    wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    wshed.at<Vec3b>(i, j) = colorTab[index - 1];
            }
        }
        if (image.dims>0)
            wshed = wshed*0.5 + image*0.5;
        return wshed;
    }

    /////////////////////////
    //  hog descriptor
    /////////////////////////

    // 计算积分图
    std::vector<Mat> CalculateIntegralHOG(Mat& srcMat)
    {
        // sobel边缘检测
        Mat sobelMatX,sobelMatY;
        Sobel(srcMat,sobelMatX,CV_32F,1,0); 
        Sobel(srcMat,sobelMatY,CV_32F,0,1);
        std::vector<Mat> bins(NBINS);
        for(int i = 0; i < NBINS; i++)
        {
            bins[i] = Mat::zeros(srcMat.size(),CV_32F);
        }
        Mat magnMat,angleMat;
        // 坐标转换
        cartToPolar(sobelMatX,sobelMatY,magnMat,angleMat,true);
        // 角度变换
        add(angleMat,Scalar(180),angleMat,angleMat<0);
        add(angleMat,Scalar(-180),angleMat,angleMat>=180);
        angleMat /= THETA;
        for(int y = 0; y < srcMat.rows;y++)
        {
            for(int x = 0; x < srcMat.cols;x++)
            {
                // 计算bins下幅值
                int ind = angleMat.at<float>(y,x);
                bins[ind].at<float>(y,x) += 
                magnMat.at<float>(y,x);
            }
        }
        // 积分图像的生成
        std::vector<Mat> integrals(NBINS);
        for(int i = 0; i < NBINS; i++)
        {
            integral(bins[i],integrals[i]);
        }
        return integrals;
    }
    // 计算单个cell HOG特征
    void cacHOGinCell(Mat& HOGCellMat, Rect roi, std::vector<Mat>& integrals)
    {
        // 快速积分HOG实现
        int x0 = roi.x,y0 = roi.y;
        int x1 = x0 + roi.width,y1 = y0 + roi.height;
        for(int i = 0; i < NBINS; i++)
        {
            // 根据矩阵的上下左右坐标
            Mat integral = integrals[i];
            float a = integral.at<double>(y0, x0);
            float b = integral.at<double>(y1, x1);
            float c = integral.at<double>(y0, x1);
            float d = integral.at<double>(y1, x0);
            HOGCellMat.at<float>(0,i) = (a + b) - (c + d);
        }
    }
    // HOG直方图获取
    cv::Mat getHog(Point pt,std::vector<Mat> &integrals)
    {
        if( pt.x - R < 0 || pt.y - R < 0 || pt.x + R >= integrals[0].cols || pt.y + R >= integrals[0].rows )
        {
            return Mat();
        }
        // 直方图
        Mat hist(Size(NBINS*BLOCKSIZE*BLOCKSIZE,1),CV_32F);
        Point tl(0,pt.y-R);
        int c = 0;
        // 遍历块
        for(int i = 0; i < BLOCKSIZE;i++)
        {
            tl.x = pt.x - R;
            for(int j = 0; j < BLOCKSIZE; j++)
            {
                // 获取当前窗口进行局部直方图计算
                Rect roi(tl,tl+Point(CELLSIZE,CELLSIZE));
                Mat hist_temp = hist.colRange(c,c+NBINS);
                cacHOGinCell(hist_temp,roi,integrals);
                tl.x += CELLSIZE;
                c += NBINS;
            }
            tl.y = CELLSIZE;
        }
        normalize(hist,hist,1,0,NORM_L2);
        return hist;
    }
    // 计算HOG特征
    std::vector<Mat> cacHOGFeature(cv::Mat srcImage)
    {
        Mat grayImage;
        std::vector<Mat> HOGMatVector;
        cv::cvtColor(srcImage,grayImage,CV_RGB2GRAY);
        grayImage.convertTo(grayImage,CV_8UC1);
        // 积分图像生成
        std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
        Mat image = grayImage.clone();
        image *= 0.5;
        // Block遍历
        cv::Mat HOGBlockMat(Size(NBINS,1),CV_32F);
        for(int y = CELLSIZE/2; y < grayImage.rows; y += CELLSIZE)
        {
            for(int x = CELLSIZE / 2; x < grayImage.cols; x += CELLSIZE)
            {
                // 获取当前窗口HOG
                cv::Mat hist = getHog(Point(x,y),integrals);
                if (hist.empty()) 
                    continue;
                HOGBlockMat = Scalar(0);
                for(int i = 0; i < NBINS; i++)
                {
                    for(int j = 0; j < BLOCKSIZE; j++)
                    {
                        HOGBlockMat.at<float>(0,i) += hist.at<float>(0,i+j*NBINS);
                    }
                }
                // L2范数归一化
                normalize(HOGBlockMat,HOGBlockMat,1,0,CV_L2);
                HOGMatVector.push_back(HOGBlockMat);    
                Point center(x, y);
                // 绘制HOG特征图
                for (int i = 0; i < NBINS; i++)
                {
                    double theta = (i * THETA + 90.0 ) * CV_PI / 180.0;
                    Point rd(CELLSIZE*0.5*cos(theta), CELLSIZE*0.5*sin(theta));
                    Point rp = center -  rd;
                    Point lp = center -  -rd;
                    line(image, rp, lp, Scalar(255*HOGBlockMat.at<float>(0, i), 255, 255));
                }
            }
        }
        imshow("out",image);
        return HOGMatVector;
    }

    /////////////////////////
    //  show histogram
    /////////////////////////

    void ShowHistogram(cv::Mat srcImage)
    {
        IplImage src2 = srcImage;
        IplImage *src = &src2;
     
        IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );  //第一个为size，第二个为位深度（8为256度），第三个通道数
        IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
        IplImage* s_plane = cvCreateImage( cvGetSize(src), 8, 1 );
        IplImage* v_plane = cvCreateImage( cvGetSize(src), 8, 1 );
        IplImage* planes[] = { h_plane, s_plane,v_plane };
     
        // / H 分量划分为16个等级，S分量划分为8个等级 
        int h_bins =16 , s_bins =8, v_bins = 8;
        int hist_size[] = {h_bins, s_bins, v_bins};
     
        //H 分量的变化范围 
        float h_ranges[] = { 0, 180 }; 
     
        //S 分量的变化范围
        float s_ranges[] = { 0, 255 };
        float v_ranges[] = { 0, 255 };

        float* ranges[] = { h_ranges, s_ranges,v_ranges};
     
        //输入图像转换到HSV颜色空间 
        cvCvtColor( src, hsv, CV_BGR2HSV );
        cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
     
        //创建直方图，二维, 每个维度上均分 
        CvHistogram * hist = cvCreateHist( 3, hist_size, CV_HIST_ARRAY, ranges, 1 );
        //根据H,S两个平面数据统计直方图 
        cvCalcHist( planes, hist, 0, 0 );
     
        //获取直方图统计的最大值，用于动态显示直方图 
        float max_value;
        cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
     
     
        //设置直方图显示图像 
        int height = 300;
        int width = (h_bins*s_bins*v_bins);
        IplImage* hist_img = cvCreateImage( cvSize(width,height), 8, 3 );
        cvZero( hist_img );
     
        //用来进行HSV到RGB颜色转换的临时单位图像 
        IplImage * hsv_color = cvCreateImage(cvSize(1,1),8,3);
        IplImage * rgb_color = cvCreateImage(cvSize(1,1),8,3);
        int bin_w = width / (h_bins * s_bins);
        for(int h = 0; h < h_bins; h++)
        {
            for(int s = 0; s < s_bins; s++)
            {
                for(int v = 0; v < v_bins; v++)
                {
                int i = h*s_bins + s*v_bins + v;
                // 获得直方图中的统计次数，计算显示在图像中的高度 
                float bin_val = cvQueryHistValue_3D( hist, h, s,v );
                int intensity = cvRound(bin_val*height/max_value);
     
                // 获得当前直方图代表的颜色，转换成RGB用于绘制 
                cvSet2D(hsv_color,0,0,cvScalar(h*180.f / h_bins,s*255.f/s_bins,v*255.f/v_bins,0));
                cvCvtColor(hsv_color,rgb_color,CV_HSV2BGR);
                CvScalar color = cvGet2D(rgb_color,0,0);
     
                cvRectangle( hist_img, cvPoint(i*bin_w,height),
                    cvPoint((i+1)*bin_w,(height - intensity)),
                    color, -1, 8, 0 );
                }
            }
        }
        // IplImage* hist_img2 = cvCreateImageHeader(cvSize(1080,580), hist_img->depth, hist_img->nChannels);
        // cvResize(hist_img, hist_img2, CV_INTER_LINEAR );
        cout<<"h "<<hist_img->height<<" w "<<hist_img->width<<endl;
        // cvNamedWindow( "H-S-V Histogram",0);
        cvShowImage( "H-S-V Histogram", hist_img);
        cvWaitKey(0);

    }


    /* ===============  histYUV  ================= */
    void ShowHistYUV(cv::Mat testImg)
    {

        int nWidth = testImg.cols, nHeight = testImg.rows;
        cout<<"testImg.cols "<<testImg.cols<<" testImg.rows "<<testImg.rows<<endl;

        cv::Mat sectionYUV(nHeight*3/2, nWidth, CV_8UC1);
        cv::cvtColor(testImg, sectionYUV, CV_BGR2YUV_I420);
        memset(sectionYUV.data, 128, nWidth*nHeight);
        cv::cvtColor(sectionYUV, testImg, CV_YUV2BGR_I420);
        imshow("rgb_back", testImg);
        long int imgHeadU = nWidth*nHeight;
        long int imgHeadV = nWidth*nHeight*5/4;

        cv::Mat plateU(nHeight/2, nWidth/2, CV_8UC1); 
        cv::Mat plateV(nHeight/2, nWidth/2, CV_8UC1);



        plateU.data = sectionYUV.data + imgHeadU;
        plateV.data = sectionYUV.data + imgHeadV;

        double maxU = 0, minU=0;
        double maxV = 0, minV=0;

        normalize(plateU, plateU, 0, 256, NORM_MINMAX, -1, Mat());
        normalize(plateV, plateV, 0, 256, NORM_MINMAX, -1, Mat());
        minMaxLoc(plateU, &minU, &maxU, 0, 0);
        minMaxLoc(plateV, &minV, &maxV, 0, 0);

        cout<<"minU "<<minU<<"maxU "<<maxU<<endl;
        
        imshow("plateU", plateU);
        imshow("plateV", plateV);
        // waitKey(0);

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
                // imshow("clrSampleRGB", clrSampleRGB);
                rectangle( histImg, Point( k*5, 512-intensity ), Point( (k+1)*5, 512 ), mean(clrSampleRGB), CV_FILLED );
                int histNum = u*(rangeUmax-rangeUmin)/histSizeU+rangeUmin;
                int histNum2 = v*(rangeVmax-rangeVmin)/histSizeV+rangeVmin;
                // if(intensity>0){
                //     cout<<"u "<<histNum<<" v "<<histNum2<<" intensity "<<intensity<<endl; 
                // }
                if(k%10==0){
                    char charText[100];
                    sprintf(charText, "%d", histNum);
                    cv::putText(histImg, charText, Point( k*5+5, 512-intensity-10 ), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 1);
                    char charText2[100];
                    sprintf(charText2, "%d", histNum2);
                    cv::putText(histImg, charText2, Point( k*5+5, 512-intensity  ), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 1);
                }
                // imshow("calcHist", histImg );
                // waitKey(0);
            }
        }


        imshow("calcHist", histImg );
        // waitKey(0);


    }


    /**/
    // void ShowHistogram(cv::Mat srcImage)
    // {
    //     IplImage src2 = srcImage;
    //     IplImage *src = &src2;
     
    //     IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );  //第一个为size，第二个为位深度（8为256度），第三个通道数
    //     IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
    //     IplImage* s_plane = cvCreateImage( cvGetSize(src), 8, 1 );
    //     IplImage* v_plane = cvCreateImage( cvGetSize(src), 8, 1 );
    //     IplImage* planes[] = { h_plane, s_plane,v_plane };
     
    //     // H 分量划分为16个等级，S分量划分为8个等级 
    //     int h_bins =16 , s_bins =8, v_bins = 8;
    //     int hist_size[] = {h_bins, s_bins};
    //     // H 分量的变化范围 
    //     float h_ranges[] = { 0, 180 }; 
     
    //     //  S 分量的变化范围
    //     float s_ranges[] = { 0, 255 };
    //     float v_ranges[] = { 0, 255 };

    //     float* ranges[] = { h_ranges, s_ranges};
     
    //     // 输入图像转换到HSV颜色空间 
    //     cvCvtColor( src, hsv, CV_BGR2HSV );
    //     cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
     
    //     //创建直方图，二维, 每个维度上均分 
    //     CvHistogram * hist = cvCreateHist( 2, hist_size, CV_HIST_ARRAY, ranges, 1 );
    //     //根据H,S两个平面数据统计直方图 
    //     cvCalcHist( planes, hist, 0, 0 );
     
    //     // 获取直方图统计的最大值，用于动态显示直方图 
    //     float max_value;
    //     cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
     
     
    //     // 设置直方图显示图像 
    //     int height = 300;
    //     int width = (h_bins*s_bins*8);
    //     IplImage* hist_img = cvCreateImage( cvSize(width,height), 8, 3 );
    //     cvZero( hist_img );

    //     //用来进行HSV到RGB颜色转换的临时单位图像 
    //     IplImage * hsv_color = cvCreateImage(cvSize(1,1),8,3);
    //     IplImage * rgb_color = cvCreateImage(cvSize(1,1),8,3);
    //     int bin_w = width / (h_bins * s_bins);
    //     for(int h = 0; h < h_bins; h++)
    //     {
    //         for(int s = 0; s < s_bins; s++)
    //         {
    //             for(int v = 0; v < v_bins; v++)
    //             {
    //             int i = h*s_bins + s;
    //             // 获得直方图中的统计次数，计算显示在图像中的高度 
    //             float bin_val = cvQueryHistValue_2D( hist, h, s );
    //             int intensity = cvRound(bin_val*height/max_value);
     
    //             // 获得当前直方图代表的颜色，转换成RGB用于绘制 
    //             cvSet2D(hsv_color,0,0,cvScalar(h*180.f / h_bins,s*255.f/s_bins,255));
    //             cvCvtColor(hsv_color,rgb_color,CV_HSV2BGR);
    //             CvScalar color = cvGet2D(rgb_color,0,0);
     
    //             cvRectangle( hist_img, cvPoint(i*bin_w,height),
    //                 cvPoint((i+1)*bin_w,height - intensity),
    //                 color, -1, 8, 0 );
    //             }
    //         }
    //     }

    //     // cvNamedWindow( "Source22", 1 );
    //     // cvShowImage( "Source22", src );
    //     cvNamedWindow( "H-S-V Histogram",1);
    //     cvShowImage( "H-S-V Histogram", hist_img );
    //     cvWaitKey(0);
    // }



























    /////////////////////////
    // sobel
    /////////////////////////

    Mat MarrEdge(Mat src)
    {
    	int kerValue = 9;
    	double delta = 1.6;
        // 计算LOG算子
        Mat kernel;
        // 半径
        int kerLen = kerValue / 2;
        kernel = Mat_<double>(kerValue, kerValue); 
        // 滑窗
        for(int i = - kerLen; i <= kerLen; i++)
        {
            for(int j = - kerLen; j <= kerLen; j++)
            {
                // 核因子生成
                kernel.at<double>(i + kerLen, j + kerLen) =
                        exp(-( ( pow(j, 2)  + pow( i, 2 )  ) /
                         ( pow(delta, 2) * 2) ))
                        * ( ( ( pow(j, 2)  + pow( i, 2 ) - 2 *
                         pow(delta, 2) ) /  (2 * pow(delta, 4) ) ));
            }
        }
        // 输出参数设置
        int kerOffset = kerValue / 2;
        Mat laplacian =  (Mat_<double>(src.rows - kerOffset * 2, 
            src.cols - kerOffset*2));
        Mat result = Mat::zeros(src.rows - kerOffset*2, 
            src.cols - kerOffset*2, src.type());
        double sumLaplacian;
        // 遍历计算卷积图像的Lapace算子
        for(int i = kerOffset; i < src.rows - kerOffset; ++i)
        {
            for(int j = kerOffset; j < src.cols - kerOffset; ++j)
            {
                sumLaplacian = 0;
                for(int k = -kerOffset; k <= kerOffset; ++k)
                {
                    for(int m = -kerOffset; m <= kerOffset; ++m)
                    {
                        // 计算图像卷积
                        sumLaplacian += src.at<uchar>(i + k, j + m) *
                         kernel.at<double>(kerOffset + k,
                         kerOffset + m);
                    }
                }
                // 生成Lapace结果
                laplacian.at<double>(i - kerOffset,
                   j - kerOffset) = sumLaplacian;
            }
        }
        // 过零点交叉 寻找边缘像素
        for(int y = 1; y < result.rows - 1; ++y)
        {
            for(int x = 1; x < result.cols - 1; ++x)
            {
                result.at<uchar>(y,x) = 0;
                // 邻域判定
                if(laplacian.at<double>(y - 1,x) * 
                    laplacian.at<double>(y + 1,x) < 0)
                {
                    result.at<uchar>(y,x) = 255;
                }
                if(laplacian.at<double>(y, x - 1) * 
                    laplacian.at<double>(y, x + 1) < 0)
                {
                    result.at<uchar>(y,x) = 255;
                }
                if(laplacian.at<double>(y + 1, x - 1) * 
                    laplacian.at<double>(y - 1, x + 1) < 0)
                {
                    result.at<uchar>(y,x) = 255;
                }
                if(laplacian.at<double>(y - 1, x - 1) *
                 laplacian.at<double>(y + 1, x + 1) < 0)
                {
                    result.at<uchar>(y,x) = 255;
                }
            }
        }
        return result;
    }

    cv::Mat GetSobel(cv::Mat srcGray)
    {
        ///imshow( "srcGray", srcGray);
        // 定义边缘图，水平及垂直
        cv::Mat edgeMat, edgeXMat, edgeYMat;
        // 求x方向Sobel边缘
        Sobel( srcGray, edgeXMat, CV_16S, 1, 0, 3, 1,
            0, BORDER_DEFAULT );
        // 求y方向Sobel边缘
        Sobel( srcGray, edgeYMat, CV_16S, 0, 1, 3, 1,
            0, BORDER_DEFAULT );
        // 线性变换转换输入数组元素成8位无符号整型
        convertScaleAbs(edgeXMat, edgeXMat);
        convertScaleAbs(edgeYMat, edgeYMat);
        // x与y方向边缘叠加
        addWeighted(edgeXMat, 0.5, edgeYMat, 0.5, 0, edgeMat);
        ///cv::imshow( "edgeYMat", edgeYMat );
        ///imshow( "edgeMat", edgeMat );
        // 定义Scharr边缘图像
        cv::Mat edgeMatS, edgeXMatS, edgeYMatS;
        // 计算x方向Scharr边缘
        Scharr( srcGray, edgeXMatS, CV_16S, 1, 0, 1,
            0, BORDER_DEFAULT );
        convertScaleAbs( edgeXMatS, edgeXMatS);
        // 计算y方向Scharr边缘
        Scharr( srcGray, edgeYMatS, CV_16S, 0, 1, 1, 
           0, BORDER_DEFAULT );
        // 线性变换转换输入数组元素成8位无符号整型
        convertScaleAbs( edgeYMatS, edgeYMatS );
        // x与y方向边缘叠加
        addWeighted( edgeXMatS, 0.5, edgeYMatS, 0.5, 0, edgeMatS );
        ///cv::imshow( "edgeMatS", edgeMatS );
        //cv::waitKey(0);
        return edgeMat;
    }


    void ShowNewSobel(cv::Mat srcGray)
    {
        //imshow( "srcGray", srcGray);
        // 定义边缘图，水平及垂直
        cv::Mat edgeMat, edgeXMat, edgeYMat;
        // 求x方向Sobel边缘
        Sobel( srcGray, edgeXMat, CV_16S, 1, 0, 3, 1,
            0, BORDER_DEFAULT );
        // 求y方向Sobel边缘
        Sobel( srcGray, edgeYMat, CV_16S, 0, 1, 3, 1,
            0, BORDER_DEFAULT );
        // 线性变换转换输入数组元素成8位无符号整型
        convertScaleAbs(edgeXMat, edgeXMat);
        convertScaleAbs(edgeYMat, edgeYMat);
        // x与y方向边缘叠加
        addWeighted(edgeXMat, 0.5, edgeYMat, 0.5, 0, edgeMat);
        cv::imshow( "edgeYMat", edgeYMat );
        imshow( "edgeMat", edgeMat );
        // 定义Scharr边缘图像
        cv::Mat edgeMatS, edgeXMatS, edgeYMatS;
        // 计算x方向Scharr边缘
        Scharr( srcGray, edgeXMatS, CV_16S, 1, 0, 1,
            0, BORDER_DEFAULT );
        convertScaleAbs( edgeXMatS, edgeXMatS);
        // 计算y方向Scharr边缘
        Scharr( srcGray, edgeYMatS, CV_16S, 0, 1, 1, 
           0, BORDER_DEFAULT );
        // 线性变换转换输入数组元素成8位无符号整型
        convertScaleAbs( edgeYMatS, edgeYMatS );
        // x与y方向边缘叠加
        addWeighted( edgeXMatS, 0.5, edgeYMatS, 0.5, 0, edgeMatS );
        cv::imshow( "edgeMatS", edgeMatS );
        cv::waitKey(0);
        cv::destroyWindow( "edgeMatS" );
        cv::destroyWindow( "edgeYMat" );
        cv::destroyWindow( "edgeMat" );
    }

    cv::Mat roberts(cv::Mat srcImage)
    {
      cv::Mat dstImage = srcImage.clone();
      int nRows = dstImage.rows;
      int nCols = dstImage.cols;
      for (int i = 0; i < nRows-1; i++)
      {
        for (int j = 0; j < nCols-1; j++)
        {
          // 根据公式计算
          int t1 = (srcImage.at<uchar>(i, j) - 
              srcImage.at<uchar>(i+1, j+1)) *
              (srcImage.at<uchar>(i, j) - 
              srcImage.at<uchar>(i+1, j+1));
          int t2 = (srcImage.at<uchar>(i+1, j) - 
              srcImage.at<uchar>(i, j+1)) *
              (srcImage.at<uchar>(i+1, j) -
               srcImage.at<uchar>(i, j+1));
          // 计算对角线像素差
          dstImage.at<uchar>(i, j) = (uchar)sqrt(t1 + t2);
        }
      }
      return dstImage;    
    }

#endif

  