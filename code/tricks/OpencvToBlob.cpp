
// 1 img.at的方式，比较慢
Mat img = imread("D:\\Deeplearning\\Caffe\\caffe\\face_examples\\mtcnn_faceRecognition\\models\\Jennifer_Aniston_0016.jpg");
vector<float>data(112 * 96 * 3,0.0);
img.convertTo(img, CV_32FC3);
int index = 0;
resize(img, img, cv::Size(width, height)); 
img = (img - 127.5)*0.0078125; //减去均值后再归一
assert(img.channels() == 3);
for (int k = 0; k < img.channels(); ++k){
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; ++j){
			index = (k*height + i)*width + j;
			data[index] = img.at<cv::Vec3f>(i, j)[k];
		}
	}
}


//2 直接取地址data 对于112*96*3的图像大约可以快2ms
vector<float>data(112 * 96 * 3, 0.0);
float* data = inputBlob->mutable_cpu_data();
	Mat img = imread("D:\\Deeplearning\\Caffe\\caffe\\face_examples\\mtcnn_faceRecognition\\models\\Jennifer_Aniston_0016.jpg");
	resize(img, img, cvSize(96, 112));
	img.convertTo(img, CV_32FC3);
	img = (img - 127.5)*0.0078125; //减去均值后再归一
	int width = inputBlob->width();
	int height = inputBlob->height();
	int spatial_dim = width*height;
	Blob<float>* inputBlob = _net->input_blobs()[0];
	float* data = inputBlob->mutable_cpu_data();
	
	assert(img.channels() == 3);
	cv::Vec3f* img_data = (cv::Vec3f*)img.data;
	for (int k = 0; k < spatial_dim; ++k){
		data[k] = (float)img_data[k][0];
		data[k + spatial_dim] = (float)img_data[k][1];
		data[k + 2 * spatial_dim] = (float)img_data[k][2];
	}


