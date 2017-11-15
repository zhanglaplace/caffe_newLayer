//**********************************************************************
// Method:   获取文件夹下所有文件
// FullName: getFiles   
// Returns:  void 
// Parameter: 输入和对应的模板路径
// Timer:2017.5.10
//**********************************************************************
void getFiles(std::string path, std::vector<std::string>  &files) {
	struct _finddata_t filefind;
	intptr_t hfile = 0;
	std::string s;
	if ((hfile = _findfirst(s.assign(path).append("/*").c_str(), &filefind)) != -1) {
		do  {
			if (filefind.attrib == _A_SUBDIR) {
				if (strcmp(filefind.name, ".") && strcmp(filefind.name, "..")){
					getFiles(s.assign(path).append("/").append(filefind.name), files);
				}
			}
			else {
				files.push_back(s.assign(path).append("/").append(filefind.name));
				std::cout << filefind.name << std::endl;
			}
		} while (_findnext(hfile, &filefind) == 0);
	} _findclose(hfile);
}
