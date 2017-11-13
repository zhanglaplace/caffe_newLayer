
// faceAnalysisDoc.cpp : CfaceAnalysisDoc 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "faceAnalysis.h"
#endif

#include "faceAnalysisDoc.h"

#include <propkey.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CfaceAnalysisDoc

IMPLEMENT_DYNCREATE(CfaceAnalysisDoc, CDocument)

BEGIN_MESSAGE_MAP(CfaceAnalysisDoc, CDocument)
	ON_COMMAND(ID_COLOR_BLACK, &CfaceAnalysisDoc::OnColorBlack)
	ON_COMMAND(ID_COLOR_RED, &CfaceAnalysisDoc::OnColorRed)
	ON_COMMAND(ID_COLOR_GREEN, &CfaceAnalysisDoc::OnColorGreen)
	ON_COMMAND(ID_COLOR_BLUE, &CfaceAnalysisDoc::OnColorBlue)
	ON_COMMAND(ID_ELEMENT_LINE, &CfaceAnalysisDoc::OnElementLine)
	ON_COMMAND(ID_ELEMENT_RECTANGLE, &CfaceAnalysisDoc::OnElementRectangle)
	ON_COMMAND(ID_ELEMENT_CIRCLE, &CfaceAnalysisDoc::OnElementCircle)
	ON_COMMAND(ID_ELEMENT_CURVE, &CfaceAnalysisDoc::OnElementCurve)
END_MESSAGE_MAP()


// CfaceAnalysisDoc 构造/析构

CfaceAnalysisDoc::CfaceAnalysisDoc()
{
	// TODO:  在此添加一次性构造代码

}

CfaceAnalysisDoc::~CfaceAnalysisDoc()
{
}

BOOL CfaceAnalysisDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO:  在此添加重新初始化代码
	// (SDI 文档将重用该文档)

	return TRUE;
}




// CfaceAnalysisDoc 序列化

void CfaceAnalysisDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO:  在此添加存储代码
	}
	else
	{
		// TODO:  在此添加加载代码
	}
}

#ifdef SHARED_HANDLERS

// 缩略图的支持
void CfaceAnalysisDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// 修改此代码以绘制文档数据
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// 搜索处理程序的支持
void CfaceAnalysisDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// 从文档数据设置搜索内容。
	// 内容部分应由“;”分隔

	// 例如:     strSearchContent = _T("point;rectangle;circle;ole object;")；
	SetSearchContent(strSearchContent);
}

void CfaceAnalysisDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = NULL;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != NULL)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CfaceAnalysisDoc 诊断

#ifdef _DEBUG
void CfaceAnalysisDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CfaceAnalysisDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CfaceAnalysisDoc 命令


void CfaceAnalysisDoc::OnColorBlack()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnColorRed()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnColorGreen()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnColorBlue()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnElementLine()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnElementRectangle()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnElementCircle()
{
	// TODO:  在此添加命令处理程序代码
}


void CfaceAnalysisDoc::OnElementCurve()
{
	// TODO:  在此添加命令处理程序代码
}
