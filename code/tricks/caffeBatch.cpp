vector<float> FaceRecognition::testBatch(vector<cv::Mat>& _vecs){

	Blob<float>* inputBlob = _net->input_blobs()[0];
	int n = _vecs.size();
	int width = inputBlob->width();
	int height = inputBlob->height();
	int nchannels = inputBlob->channels();
	inputBlob->Reshape(n, nchannels, height, width);
	_net->Reshape();
	int nums = inputBlob->num();

	for (int i = 0; i < n; ++i){
		_vecs[i].convertTo(_vecs[i], CV_32FC3);
		resize(_vecs[i], _vecs[i], cv::Size(width, height));
		_vecs[i] = (_vecs[i] - 127.5)*0.0078125; //减去均值后再归一
	}

	vector<cv::Mat>im = _vecs;
	assert(im.size(), nums);
	int index = 0;
	float* data = inputBlob->mutable_cpu_data();

	for (int n = 0; n < nums; ++n){
		for (int k = 0; k < nchannels; ++k){
			for (int i = 0; i < height; ++i){
				for (int j = 0; j < width; ++j){
					index = ((n* nchannels + k)*height + i)*width + j;
					data[index] = im[n].at<cv::Vec3f>(i, j)[k];
				}
			}
		}
	}
	Blob<float>* outputBlob = _net->Forward()[0];
	const float* begin = outputBlob->cpu_data();
	const float* end = begin + nums*outputBlob->channels();
	return vector<float>(begin, end);

}


vector<float> FaceRecognition::testSingle(cv::Mat _vecs){

	Blob<float>* inputBlob = _net->input_blobs()[0];
	int width = inputBlob->width();
	int height = inputBlob->height();
	int nchannels = inputBlob->channels();
	int nums = inputBlob->num();

	
	_vecs.convertTo(_vecs, CV_32FC3);
	resize(_vecs, _vecs, cv::Size(width, height));
	_vecs = (_vecs - 127.5)*0.0078125; //减去均值后再归一


	cv::Mat im = _vecs.clone();
	assert(im.size(), nums);
	int index = 0;
	float* data = inputBlob->mutable_cpu_data();
	int spatial_size = height*width;
	cv::Vec3f * img_data = (cv::Vec3f *)im.data;
	for (int k = 0; k < spatial_size; ++k){
		data[k] = float(img_data[k][0]);
		data[k + spatial_size] = float(img_data[k][1]);
		data[k + 2 * spatial_size] = float(img_data[k][2]);
	}

	Blob<float>* outputBlob = _net->Forward()[0];
	const float* begin = outputBlob->cpu_data();
	const float* end = begin + nums*outputBlob->channels();
	return vector<float>(begin, end);

}
