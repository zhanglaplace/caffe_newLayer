#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <vector>
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
  ifstream fin("F:\\datasets\\facial_expression\\AffectNet\\ImageSets\\Main\\test.txt",ios_base::in);
  ofstream fout("F:\\datasets\\facial_expression\\AffectNet\\ImageSets\\Main\\test_voc.txt",ios_base::out);
  string lines;
  int count = 0;
  while(getline(fin,lines)){
    vector<string>vc;
    SplitString(lines,vc,".");
    fout << vc[0] <<endl;
    count++;
    if (count % 1000 == 0) {
      cout<<count<<endl;
    }
  }
  fin.close();
  fout.close();
  return 0;
}
