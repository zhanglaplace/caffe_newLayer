#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#define TYPECLASS 7

vector<string>facial_expression{"Neutral","Happiness","Sadness","Surprise","Fear","Disgust","Anger"};

/******************************************************
// SplitString
// 分割字符串
// 张峰
// 2017.07.26
/*******************************************************/
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

/******************************************************
// main
// 解析affectNet的train 和val .csv 保存格式为  filename class_type face_x face_y face_w face_h
// 张峰
// 2017.07.26
/*******************************************************/
int main(void){
  const string dirName = "F:/datasets/facial_expression/AffectNet/Manually_Annotated_file_lists/validation.csv";
  const string dstName = "F:/datasets/facial_expression/AffectNet/Manually_Annotated_file_lists/validation_1.txt";
  ifstream fin(dirName.c_str(),ios_base::in);
  ofstream fou(dstName.c_str(),ios_base::out);
  if (!fin) {
    cout<<"open file"<< dirName <<"error\n";
    return -1;
  }

  string lines;
  int count = 0; // 统计样本个数，每隔1000个样本打印一次
  int class_type[7] = {0};
  getline(fin,lines);//第一行不需要
  while(getline(fin,lines)){
      vector<string>line,name;
      SplitString(lines,line,",");
      SplitString(line[0],name,"/");
      if(line.size() != 9){
        cout<<"phase "<<lines<<"error\n";
        return -1;
      }
      // 只考虑7种基本的表情
      if (atoi(line[6].c_str()) >= 7 ) {
        continue;
      }
      count++; // 有效的表情数量+1
      fou<<line[0]<<" "<<facial_expression[atoi(line[6].c_str())]<<" "<<line[1]<<" "<<line[2]<<" "<<atoi(line[3].c_str())+atoi(line[1].c_str())<<" "<<atoi(line[4].c_str())+atoi(line[2].c_str())<<endl;
      if (count%1000 == 0) {
        cout<<"finishing "<< count<<" images..........\n";
      }
      class_type[atoi(line[6].c_str())]++;
  }
  fou.close();
  fin.close();

  cout<<"total images:" << count<<endl;
  for (size_t i = 0; i < TYPECLASS; i++) {
    cout<< facial_expression[i]<<": "<< class_type[i]<<endl;
  }
  return 0;
}
