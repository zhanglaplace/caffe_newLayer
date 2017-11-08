---
title: MFC学习
date: 2017-11-08 09:03:49
tags: [MFC]
categories: MFC
---

# MFC学习
  慢慢积累$MFC$的知识,多做总结

## 异常
  (1)$MFC$抛出的异常通常为$CDBException$,
```cpp
  try{
     // execute some code
  }
  catch(CException* ex){
    // Handle the exception here...
  }
```
  (2) 处理内存分配错误
  通常内存分配，比如new分配内存失败时候，程序应当终止，但是也可以通过捕获new异常，进行相应处理。
```cpp
  size_t count = ~static_cast<size_t>(0)/2;
  try{
    char* data = new char[count];
    cout<<"Memory allocated.\n";
  }
  catch(bad_alloc& ex){
    cout<<"Memory allocation failed.\n"<<"The information from the exception object is:"
    << ex.what()<<endl;
  }
  delete[] data;
```

##
