#include <windows.h>


LARGE_INTEGER clockFreq;
LARGE_INTEGER startTime;
LARGE_INTEGER endTime;
double elapsedTime = 0.0;
QueryPerformanceFrequency(&clockFreq);
QueryPerformanceCounter(&startTime);
// your code
QueryPerformanceCounter(&endTime);
elapsedTime += (static_cast<double>(endTime.QuadPart - startTime.QuadPart) / clockFreq.QuadPart);
std::cout << elapsedTime * 1000.0 << " ms" << std::endl;

//opencv
double t = (double)cv::getTickCount();
// your code
std::cout <<  " time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s" << std::endl	

// chrone
std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
// your code
std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;



// ctime
clock_t t = clock();
// your code
cout << "detection time:" << clock() - t << "ms" << endl;
