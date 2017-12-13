  cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv = Mat()) {
    assert(source_points.size() == target_points.size());
    assert(source_points.size() >= 2);
    Mat U = Mat::zeros(target_points.size() * 2, 1, CV_64F);
    Mat X = Mat::zeros(source_points.size() * 2, 4, CV_64F);
    for (int i = 0; i < target_points.size(); i++) {
      U.at<double>(i * 2, 0) = source_points[i].x;
      U.at<double>(i * 2 + 1, 0) = source_points[i].y;
      X.at<double>(i * 2, 0) = target_points[i].x;
      X.at<double>(i * 2, 1) = target_points[i].y;
      X.at<double>(i * 2, 2) = 1;
      X.at<double>(i * 2, 3) = 0;
      X.at<double>(i * 2 + 1, 0) = target_points[i].y;
      X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
      X.at<double>(i * 2 + 1, 2) = 0;
      X.at<double>(i * 2 + 1, 3) = 1;
    }
    Mat r = X.inv(DECOMP_SVD)*U;
    Tinv = (Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
                         r.at<double>(1), r.at<double>(0), 0,
                         r.at<double>(2), r.at<double>(3), 1);
    Mat T = Tinv.inv(DECOMP_SVD);
    Tinv = Tinv(Rect(0, 0, 2, 3)).t();
    return T(Rect(0,0,2,3)).t();
  }

  cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv = Mat()) {
    Mat Tinv1, Tinv2;
    Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
    std::vector<Point2d> source_point_reflect;
    for (auto sp : source_points) {
      source_point_reflect.push_back(Point2d(-sp.x, sp.y));
    }
    Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
    trans2.colRange(0,1) *= -1;
    std::vector<Point2d> trans_points1, trans_points2;
    transform(source_points, trans_points1, trans1);
    transform(source_points, trans_points2, trans2);
    double norm1 = norm(Mat(trans_points1), Mat(target_points), NORM_L2);
    double norm2 = norm(Mat(trans_points2), Mat(target_points), NORM_L2);
    Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
    return norm1 < norm2 ? trans1 : trans2;
  }
  
  std::vector<cv::Point2d> getVertexFromBox(Rect2d box) {
    return{ Point2d(box.x, box.y), Point2d(box.x + box.width, box.y), Point2d(box.x + box.width, box.y + box.height), Point2d(box.x, box.y + box.height) };
  }
  
  
  vector<Point2d> target_points = { {30.2946,51.6963},{65.5318,51.5014},{48.0252,71.7366},{33.5493,92.3655},{62.7299,92.2041} };
  // 68 点对应五点：
    left_eye_x = 0, left_eye_y = 0;
	right_eye_x = 0, right_eye_y = 0;
	for (int i = 36; i < 42 ; i++){
		left_eye_x += shape.at<double>(i, 0);
		left_eye_y += shape.at<double>(i, 1);
	}
	for (int i = 42; i < 48; i++)
	{
		right_eye_x += shape.at<double>(i, 0);
		right_eye_y += shape.at<double>(i, 1);
	}

	left_eye_x = left_eye_x/ 6.0 ;
	left_eye_y = left_eye_y / 6.0 ;
	right_eye_x = right_eye_x / 6.0 ;
	right_eye_y = right_eye_y / 6.0; 
	
	points.clear();
	points.push_back(Point2d(left_eye_x,left_eye_y));
	points.push_back(Point2d(right_eye_x,right_eye_y));
	points.push_back(Point2d(shape.at<double>(30,0),shape.at<double>(30,1)));
	points.push_back(Point2d(shape.at<double>(48,0),shape.at<double>(48,1)));
	points.push_back(Point2d(shape.at<double>(54,0),shape.at<double>(54,1)))
	
	
  // 68 点对应五点。。
  
  Mat trans_inv，cropImage;
  Mat trans = findSimilarityTransform(points, target_points, trans_inv);
  warpAffine(image, cropImage, trans, Size(96, 112));
  // 画出斜的矩形
  vector<Point2d> rotatedVertex;
  transform(getVertexFromBox(Rect(0,0,96,112)), rotatedVertex, trans_inv);
  for (int i = 0; i < 4; i++)
	line(show_image, rotatedVertex[i], rotatedVertex[(i + 1) % 4], Scalar(0, 255, 0), 2);
  
  
  
  
  
  