/******************************************************
// 函数名: 
// 说明:
// 作者:张峰
// 时间:2017.12.09
// 备注:测试图像加上角度旋转后，特征点旋转是否正常
/*******************************************************/
int testlandmarkwarp(float max_angle){

	string pts_file = "D:/others/FacialExpressionImage/facial_landmark/lfpw/trainset/image_0011.pts";
	FILE *tmp = fopen(pts_file.c_str(), "r");
	string img_file = pts_file.substr(0, pts_file.find_last_of('.')) + ".png";
	Mat img = imread(img_file);

	char line[256];
	Mat_<float> shape(68, 2);
	fgets(line, sizeof(line), tmp);
	fgets(line, sizeof(line), tmp);
	fgets(line, sizeof(line), tmp);
	float max_x = 0, max_y = 0, min_x = img.cols, min_y = img.rows;
	for (int  i = 0; i < 68; i++){
		fscanf(tmp, "%f", &shape(i,0));
		fscanf(tmp, "%f",&shape(i, 1));
		circle(img, Point(shape(i, 0), shape(i, 1)), 2, cv::Scalar(255, 0, 0));
		if (shape(i,0) > max_x){
			max_x = shape(i, 0);
		}
		if (shape(i, 1) > max_y){
			max_y = shape(i, 1);
		}
		if (shape(i, 0) < min_x){
			min_x = shape(i, 0);
		}
		if (shape(i, 1) < min_y){
			min_y = shape(i, 1);
		}
	}
	fclose(tmp);
	float center_x = (min_x + max_x) / 2.0;
	float center_y = (min_y + max_y) / 2.0;


	RNG rng(getTickCount());
	float alpha = rng.uniform(-max_angle, max_angle);

	Mat roi = getRotationMatrix2D(Point(center_x, center_y), alpha, 1);
	Mat dst = img.clone();
 	warpAffine(img, dst, roi, cv::Size(img.cols, img.rows), 1, 4);
	for (int i = 0; i < 68; i++){
		Point2f p(0.f, 0.f);
		p.x = roi.ptr<float>(0)[0] * shape(i, 0) + roi.ptr<float>(0)[1] * shape(i, 1) + roi.ptr<float>(0)[1];
		p.y = roi.ptr<float>(1)[0] * shape(i, 0) + roi.ptr<float>(1)[1] * shape(i, 1) + roi.ptr<float>(1)[2];
		circle(dst, p,2, cv::Scalar(0, 255, 0));

	}
	return 1;

}