
// faceAnalysis.h : faceAnalysis Ӧ�ó������ͷ�ļ�
//
#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"       // ������


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
	virtual int ExitInstance();

// ʵ��
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CfaceAnalysisApp theApp;
