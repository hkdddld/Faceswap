#ifndef _renderFace_H_// 查看 _renderFace_H_是否定义过来判断有无包含这个头文件如果包含就跳过

#define _renderFace_H_

using namespace cv; 
using namespace std; 
vector<vector<Point2f>> detectfaceandlandmarks(Mat& inputFrame, CascadeClassifier& faceDetector, cv::Ptr<cv::face::Facemark> facemark) {
    Mat gray;
    vector<Rect> faces;
    cvtColor(inputFrame, gray, COLOR_BGR2GRAY);
    faceDetector.detectMultiScale(gray, faces);//检测出人脸区域存在faces中
    vector< vector<Point2f>> landmarks;
    bool success = facemark->fit(inputFrame, faces, landmarks);
    return landmarks;
    
}

vector<Point2f> readPoints(string pointsFileName){
	vector<Point2f> points;
	ifstream ifs (pointsFileName.c_str());//表示返回一个（c风格）指向空字符结尾的字符数组的指针
    float x, y;
	int count = 0;
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x,y));

    }

	return points;
}

// 仿射变换
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
    // 给定一对三角形，找到仿射变换。
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    
    // 将刚刚找到的仿射变换应用于 src 图像
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}


// 计算点集的 Delaunay 三角形
//返回每个三角形的 3 个点的索引向量
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri){

	//生成rect大小的矩阵
    Subdiv2D subdiv(rect);

	// Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);//将点插入矩阵	        

	vector<Vec6f> triangleList;//存储了6个浮点数的向量
	subdiv.getTriangleList(triangleList);//将计算出来的三角形列表存储到triangleList
	vector<Point2f> pt(3);
	vector<int> ind(3);

	for( size_t i = 0; i < triangleList.size(); i++ )//遍历每个三角形
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5 ]);

		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){//判断三角形在圈内
			for(int j = 0; j < 3; j++)//三角形的点
				for(size_t k = 0; k < points.size(); k++)//遍历图像中的点如找到跟图像误差在1内的点  因为计算出来包含小数所以用近似
					if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)						
						ind[j] = k;					

			delaunayTri.push_back(ind);
		}
	}
		
}


//这样就实现了将三角形 t1 和 t2 从 img1 中映射到 img2 中，并通过 alpha 混合实现无缝融合效果。 切割出目标脸部
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2)
{
    
    Rect r1 = boundingRect(t1);//计算出包围t1的最小矩形
    Rect r2 = boundingRect(t2);
    
    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect;
    vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {
// 这样做的目的是为了将三角形的顶点坐标转换为相对于它们各自外接矩形的局部坐标系。 方便仿射变换计算 
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly

    }
    
    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);//用白色填充 t2RectInt多边形顶点
    
    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    img1(r1).copyTo(img1Rect);
    
    Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());
    
    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
    //multiply 更多地用于图像的像素级乘法操作 图像混合
    multiply(img2Rect,mask, img2Rect);//mask指定需要混合的区域
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));//(1.0, 1.0, 1.0) - mask 则用于指定哪些区域不需要融合。
    img2(r2) = img2(r2) + img2Rect;
    
    
}
#endif // _renderFace_H_ 跳过部分到这里结束