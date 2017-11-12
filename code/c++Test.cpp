#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <vector>
#include <iomanip>
#include <map>
using namespace std;


void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}



int main(void){
  ifstream fin("D:\\Deeplearning\\Caffe\\caffe\\face_examples\\mtcnn_faceRecognition\\matlab\\sphereface_flip_cos_fliplayer.txt",ios_base::in);
  ofstream fout("D:\\Deeplearning\\Caffe\\caffe\\face_examples\\mtcnn_faceRecognition\\matlab\\sphereface_flip_cos_fliplayer_final.txt",ios_base::out);
  string lines;
  int count = 0;
	cout<<fixed<<setprecision(6)<<0.2<<endl;
  while(getline(fin,lines)){
    vector<string>vc;
    SplitString(lines,vc," ");
    count++;
		fout<<vc[0]<<" "<<setprecision(6)<<vc[1]<<endl;
		if (count %100 == 0) {
			cout<<count<<endl;
		}
  }
  fin.close();
	fout.close();
  return 0;
}
