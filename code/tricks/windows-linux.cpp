// linux
#include <sys/stat.h>
mkdir(dst_dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
// windows 
#include <dirent>
_mkdir(dst_dir.c_str());
