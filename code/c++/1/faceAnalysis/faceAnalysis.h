
// faceAnalysis.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CfaceAnalysisApp: 
// �йش����ʵ�֣������ faceAnalysis.cpp
//

class CfaceAnalysisApp : public CWinApp
{
public:
	CfaceAnalysisApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CfaceAnalysisApp theApp;