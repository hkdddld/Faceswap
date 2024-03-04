#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <string> 
#include "faceswap.hpp"
using namespace cv;
using namespace std;
using namespace cv::face;

int main( int argc, char** argv)
{	
    //载入人脸检测模型
    CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
    // 创建Facemark 实例
    Ptr<Facemark> facemark = FacemarkLBF::create();
    // 加载人脸关键点检测模型
    facemark->loadModel("lbfmodel.yaml");

      

    string filename1 = "204431.jpg";
    string filename2 = "222203.jpg";
     vector< vector<Point2f>> landmarks_1,landmarks_2;
    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    Mat img1Warped = img2.clone();
    landmarks_1= detectfaceandlandmarks(img1,faceDetector, facemark);
    landmarks_2= detectfaceandlandmarks(img2,faceDetector, facemark);
    
    vector<Point2f> points1, points2;
    if (!landmarks_1.empty()) {//向量是否为空
    points1.assign(landmarks_1[0].begin(), landmarks_1[0].end());//assign复制 
    points2.assign(landmarks_2[0].begin(), landmarks_2[0].end());
          }
    else{
      cerr << "Error: Landmarks vector is empty." << endl;
      return -1;
    }
  
    
    
    img1.convertTo(img1, CV_32F);
    img1Warped.convertTo(img1Warped, CV_32F);
    
    
    
    vector<Point2f> hull1;
    vector<Point2f> hull2;
    vector<int> hullIndex;
    
    convexHull(points2, hullIndex, false, false);
    
    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
    }

    
   
    vector< vector<int> > dt;
    Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
    calculateDelaunayTriangles(rect, hull2, dt);
    
    // Apply affine transformation to Delaunay triangles
    for(size_t i = 0; i < dt.size(); i++)
    {
        vector<Point2f> t1, t2;
     
        for(size_t j = 0; j < 3; j++)
        {
            t1.push_back(hull1[dt[i][j]]);
            t2.push_back(hull2[dt[i][j]]);
        }
        
        warpTriangle(img1, img1Warped, t1, t2);

    }
    imshow("Face Swapped-1", img1Warped/255.0);
    // Calculate mask
    vector<Point> hull8U;
    for(int i = 0; i < hull2.size(); i++)
    {
        Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
    fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));

    // Clone seamlessly.
    Rect r = boundingRect(hull2);
    Point center = (r.tl() + r.br()) / 2;
    
    Mat output;
    img1Warped.convertTo(img1Warped, CV_8UC3);
    seamlessClone(img1Warped,img2, mask, center, output, NORMAL_CLONE);
    
    imshow("Face Swapped", output);
    waitKey(0);
	  return 0;
}