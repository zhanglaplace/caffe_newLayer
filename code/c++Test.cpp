#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <vector>
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

	map<string, int> IntHash;
	IntHash["Neutral"] = 0;
	IntHash["Happiness"] = 1;
	IntHash["Sadness"] = 2;
	IntHash["Surprise"] = 3;
	IntHash["Fear"] = 4;
	IntHash["Disgust"] = 5;
	IntHash["Anger"] = 6;
  ifstream fin("D:\\Deeplearning\\Caffe-ssd-Microsft\\caffe\\face_examples\\SSD\\training.txt",ios_base::in);
  ofstream fout("D:\\Deeplearning\\Caffe-ssd-Microsft\\caffe\\face_examples\\SSD\\train_list.txt",ios_base::out);
	ofstream fouv("D:\\Deeplearning\\Caffe-ssd-Microsft\\caffe\\face_examples\\SSD\\validate_list.txt",ios_base::out);
	const string src_path = "E:/deeplearning/caffe/data/VOC0712/JPEGImages/";
  string lines;
  int count = 0;
  while(getline(fin,lines)){
    vector<string>vc;
    SplitString(lines,vc," ");
    count++;
    if (count % 1000 == 0) {
      cout<<count<<endl;
    }
		if (count < 250000) {
			fout << src_path + vc[0]<<" "<<IntHash[vc[1]]<<endl;
		}
		else{
			fouv << src_path + vc[0]<<" "<<IntHash[vc[1]]<<endl;
		}
  }
  fin.close();
  fout.close();
	fouv.close();
  return 0;
}
