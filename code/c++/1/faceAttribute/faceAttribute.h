
// faceAttribute.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CfaceAttributeApp: 
// �йش����ʵ�֣������ faceAttribute.cpp
//

class CfaceAttributeApp : public CWinApp
{
public:
	CfaceAttributeApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CfaceAttributeApp theApp;