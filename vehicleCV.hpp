#ifndef _VEHICLECV_
#define _VEHICLECV_

#include <stdio.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv.hpp>
#include <sys/time.h>

#include <math.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <vector>

#include "debug.hpp"

#define CAR_CASCADE1 "cascade_vehicle1.xml"
#define CAR_CASCADE2 "cascade_vehicle2.xml"

using namespace cv;
using namespace std;

///the class to use opencv detection interface
class VEHCV : public CascadeClassifier
{
	public:
	void FixScaleDet( const Mat& image, double factor, vector<Rect>& candidates);
	void MultiScaleDet( const Mat& image, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels );
};

class CvDetection
{
public:
    VEHCV VehCascade;
    bool classifierLoadSuccess;
public:
    void LoadClassifier(string classifierName);
    std::vector<cv::Rect> DetectVehicle(cv::Mat imgInput, double scalseFactor);
    void GroupRectangles(std::vector<cv::Rect>& rectList, int groupThreshold, double eps);
    void GroupRectangles(std::vector<cv::Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights);
};

class CV_EXPORTS SameRects
{
public:
    SameRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
        std::abs(r1.y - r2.y) <= delta &&
        std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
        std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
        printf("%d %d %d %d\n", r1.x,r1.y,r1.x + r1.width,r1.y + r1.height);
        printf("%d %d %d %d\n", r2.x,r2.y,r2.x + r2.width,r2.y + r2.height);
        cout<<"eps "<<eps<<endl;
    }
    double eps;
};

template<typename _Tp, class _EqPredicate> int
Partition( const vector<_Tp>& _vec, vector<int>& labels,
           _EqPredicate predicate=_EqPredicate())
{
    int i, j, N = (int)_vec.size();///detect rect number
    const _Tp* vec = &_vec[0];

    const int PARENT=0;
    const int RANK=1;

    vector<int> _nodes(N*2);
    int (*nodes)[2] = (int(*)[2])&_nodes[0];

    // The first O(N) pass: create N single-vertex trees
    for(i = 0; i < N; i++)
    {
        nodes[i][PARENT]=-1;
        nodes[i][RANK] = 0;
    }

    // The main O(N^2) pass: merge connected components
    for( i = 0; i < N; i++ )
    {
        int root = i;

        // find root
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];

        for( j = 0; j < N; j++ )
        {
        	
            if( i == j || !predicate(vec[i], vec[j]))
                continue;
            int root2 = j;
            //cout<<"predicate(vec["<<i<<"], vec["<<j<<"]) "<<predicate(vec[i], vec[j])<<endl;
            while( nodes[root2][PARENT] >= 0 )
                root2 = nodes[root2][PARENT];

            if( root2 != root )
            {
                // unite both trees
                int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                if( rank > rank2 )
                    nodes[root2][PARENT] = root;
                else
                {
                    nodes[root][PARENT] = root2;
                    nodes[root2][RANK] += rank == rank2;
                    root = root2;
                }
                assert( nodes[root][PARENT] < 0 );

                int k = j, parent;

                // compress the path from node2 to root
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }

                // compress the path from node to root
                k = i;
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
            }
        }
    }

    // Final O(N) pass: enumerate classes
    labels.resize(N);
    int nclasses = 0;

    for( i = 0; i < N; i++ )
    {
        int root = i;
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];
        // re-use the rank as the class label
        if( nodes[root][RANK] >= 0 )
            nodes[root][RANK] = ~nclasses++;
        labels[i] = ~nodes[root][RANK];
        //cout<<"labels["<<i<<"] "<<labels[i]<<endl;

    }

    return nclasses;
}

#endif