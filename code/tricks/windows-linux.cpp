// linux
#include <sys/stat.h>
#include <unistd.h>
mkdir(dst_dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
access
// windows 
#include <dirent>
#include <direct.h>
#include <io.h>
_mkdir(dst_dir.c_str());
_access(dst_dir.c_str(),0) == -1;
