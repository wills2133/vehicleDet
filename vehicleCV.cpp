#include "vehicleCV.hpp"


void CvDetection::LoadClassifier(string classifier)
{
    classifierLoadSuccess = VehCascade.load(classifier);
    if(classifierLoadSuccess)
        cout<<"load cascade file: "<<classifier<<endl;
    if(!classifierLoadSuccess)
        cout<<"Could not load cascade file: "<<classifier<<endl;
}

vector<Rect> CvDetection::DetectVehicle(Mat imgInput, double scalseFactor)
{
    vector<Rect> rectVehs;
    if(!imgInput.data){
        std::cout<<"failed to load roiImg"<<std::endl;
    }
         
//T4
    #if DEBUG 
        if(doMutiScaleDet)
        {
            printf("MultiScaleDet\n");
            //VehCascade.detectMultiScale(imgInput, cars, 1.1, 1, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
            VehCascade.MultiScaleDet(imgInput, rectVehs, 1.05, 1, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(), false  );
        }
        else
    #endif
        {
            resize( imgInput, imgInput, cv::Size(), scalseFactor, scalseFactor, CV_INTER_LINEAR  );
            VehCascade.FixScaleDet( imgInput,  1/scalseFactor , rectVehs);
        }
//T5                
    if(0){
        int minNeighbors = 1;
        const double GROUP_EPS = 0.2;
        GroupRectangles( rectVehs, minNeighbors, GROUP_EPS );
        cout<<"doNMS"<<endl;
    }


    return rectVehs;
}

///redifine
///void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights)
///remove the overlapped detected rectangluar
void CvDetection::GroupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights)
{
    if( groupThreshold < 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }

    vector<int> labels;
    int nclasses = Partition(rectList, labels, SameRects(eps));
    //cout<<"nclasses "<<nclasses<<endl;
    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
        //cout<<"rectList["<<i<<"].x "<<rectList[i].x<<endl;
    }
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        //cout<<"rrects["<<i<<"].x "<<rrects[i].x<<endl;
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();
    //cout<<"nclasses2 "<<nclasses<<endl;
    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        //cout<<"jrrects["<<i<<"].x "<<rrects[i].x<<endl;
        int n1 = levelWeights ? rejectLevels[i] : rweights[i];
        double w1 = rejectWeights[i];
        //cout<<"levelWeights "<<levelWeights<<" rejectLevels[i] "<<rejectLevels[i]<<" rweights[i] "<<rweights[i]<<endl;
        //cout<<"n1 "<<n1<<"groupThreshold "<<groupThreshold<<endl;
        if( n1 < groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            //cout<<"r1.x "<<r1.x<<endl;
            rectList.push_back(r1);
            if( weights )
                weights->push_back(n1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

///redifine
///void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
///remove the overlapped detected rectangluar
void CvDetection::GroupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
    GroupRectangles(rectList, groupThreshold, eps, 0, 0);
}


///redefine 
///bool CascadeClassifier::detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                           // int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                           // vector<int>& levels, vector<double>& weights, bool outputRejectLevels )
///detect img in one scale
void VEHCV::FixScaleDet( const Mat& image, double factor, vector<Rect>& candidates)
{
    Size originalWindowSize = getOriginalWindowSize();
    //cout<<"originalWindowSize.width "<<originalWindowSize.width<< " originalWindowSize.height "<<originalWindowSize.height<<endl;
    Size processingRectSize(image.cols-originalWindowSize.width+1, image.rows-originalWindowSize.height+1);
    int stripSize = processingRectSize.height;
    int stripCount = 1;
    //cout<<"VehCascade.stripSize "<<stripSize<<endl;
    int yStep = 4;     
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    //#ifdef HAVE_TBB
        //std::cout<<"HAVE_TBB"<<std::endl;
    //#endif
    if(detectSingleScale( image, stripCount, processingRectSize, stripSize, 
        yStep, factor, candidates, fakeLevels, fakeWeights, false ) ){}
    //  std::cout<<"detect vehicles"<<std::endl;
    //else
        //std::cout<<"detect nothings"<<std::endl;
}

///redefine 
///void CascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                          // vector<int>& rejectLevels,
                                          // vector<double>& levelWeights,
                                          // double scaleFactor, int minNeighbors,
                                          // int flags, Size minObjectSize, Size maxObjectSize,
                                          // bool outputRejectLeve
///detect img in different scales
void VEHCV::MultiScaleDet( const Mat& image, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    
    ///
    vector<int> rejectLevels;
    vector<double> levelWeights;
    ///

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;

    objects.clear();

    if (!maskGenerator.empty()) {
        maskGenerator->initializeMask(image);
    }


    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();

    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }

    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
    vector<Rect> candidates;

    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size originalWindowSize = getOriginalWindowSize();

        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
        Size processingRectSize( scaledImageSize.width - originalWindowSize.width + 1, scaledImageSize.height - originalWindowSize.height + 1 );

        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;

        Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

        int yStep;
        if( getFeatureType() == cv::FeatureEvaluator::HOG )
        {
            yStep = 4;
        }
        else
        {
            yStep = factor > 2. ? 1 : 2;
        }

        int stripCount, stripSize;

    #ifdef HAVE_TBB
        const int PTS_PER_THREAD = 1000;
        stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;
    #else
        stripCount = 1;
        stripSize = processingRectSize.height;
    #endif

        if( !detectSingleScale( scaledImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates,
            rejectLevels, levelWeights, outputRejectLevels ) )
            break;
    }


    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());
    

    const double GROUP_EPS = 0.2;
    if( outputRejectLevels )
    {
        groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
    }
    else
    {
        groupRectangles( objects, minNeighbors, GROUP_EPS );
    }
    
}