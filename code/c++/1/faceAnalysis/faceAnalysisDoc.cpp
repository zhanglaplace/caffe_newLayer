
// faceAnalysisDoc.cpp : CfaceAnalysisDoc ���ʵ��
//

#include "stdafx.h"
// SHARED_HANDLERS ������ʵ��Ԥ��������ͼ������ɸѡ�������
// ATL ��Ŀ�н��ж��壬�����������Ŀ�����ĵ����롣
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
	ON_UPDATE_COMMAND_UI(ID_COLOR_BLACK, &CfaceAnalysisDoc::OnUpdateColorBlack)
	ON_UPDATE_COMMAND_UI(ID_COLOR_RED, &CfaceAnalysisDoc::OnUpdateColorRed)
	ON_UPDATE_COMMAND_UI(ID_COLOR_GREEN, &CfaceAnalysisDoc::OnUpdateColorGreen)
	ON_UPDATE_COMMAND_UI(ID_COLOR_BLUE, &CfaceAnalysisDoc::OnUpdateColorBlue)
	ON_UPDATE_COMMAND_UI(ID_ELEMENT_LINE, &CfaceAnalysisDoc::OnUpdateElementLine)
	ON_UPDATE_COMMAND_UI(ID_ELEMENT_RECTANGLE, &CfaceAnalysisDoc::OnUpdateElementRectangle)
	ON_UPDATE_COMMAND_UI(ID_ELEMENT_CIRCLE, &CfaceAnalysisDoc::OnUpdateElementCircle)
	ON_UPDATE_COMMAND_UI(ID_ELEMENT_CURVE, &CfaceAnalysisDoc::OnUpdateElementCurve)
END_MESSAGE_MAP()


// CfaceAnalysisDoc ����/����

CfaceAnalysisDoc::CfaceAnalysisDoc()
{
	// TODO:  �ڴ����һ���Թ������

}

CfaceAnalysisDoc::~CfaceAnalysisDoc()
{
}

BOOL CfaceAnalysisDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO:  �ڴ�������³�ʼ������
	// (SDI �ĵ������ø��ĵ�)

	return TRUE;
}




// CfaceAnalysisDoc ���л�

void CfaceAnalysisDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO:  �ڴ���Ӵ洢����
	}
	else
	{
		// TODO:  �ڴ���Ӽ��ش���
	}
}

#ifdef SHARED_HANDLERS

// ����ͼ��֧��
void CfaceAnalysisDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// �޸Ĵ˴����Ի����ĵ�����
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

// ������������֧��
void CfaceAnalysisDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// ���ĵ����������������ݡ�
	// ���ݲ���Ӧ�ɡ�;���ָ�

	// ����:     strSearchContent = _T("point;rectangle;circle;ole object;")��
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

// CfaceAnalysisDoc ���

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


// CfaceAnalysisDoc ����


void CfaceAnalysisDoc::OnColorBlack()
{
	// TODO:  �ڴ���������������
	m_Color = ElementColor::BLACK; // ���û�����ɫΪ��ɫ
}


void CfaceAnalysisDoc::OnColorRed()
{
	// TODO:  �ڴ���������������
	m_Color = ElementColor::RED; // ���û�����ɫΪ��ɫ
}


void CfaceAnalysisDoc::OnColorGreen()
{
	// TODO:  �ڴ���������������
	m_Color = ElementColor::GREEN; // ���û�����ɫΪ��ɫ
}


void CfaceAnalysisDoc::OnColorBlue()
{
	// TODO:  �ڴ���������������
	m_Color = ElementColor::BLUE; // ���û�����ɫΪ��ɫ
}


void CfaceAnalysisDoc::OnElementLine()
{
	// TODO:  �ڴ���������������
	m_Element = ElementType::LINE; // ���û�����״Ϊ����
}


void CfaceAnalysisDoc::OnElementRectangle()
{
	// TODO:  �ڴ���������������
	m_Element = ElementType::RECTANGLE; //���û�����״Ϊ����
}


void CfaceAnalysisDoc::OnElementCircle()
{
	// TODO:  �ڴ���������������
	m_Element = ElementType::CIRCLE; //���û�����״ΪԲ��
}


void CfaceAnalysisDoc::OnElementCurve()
{
	// TODO:  �ڴ���������������
	m_Element = ElementType::CURVE; //���û�����״Ϊ����
}


// ֻ�������´�������MFC��,�������ڹ������Ͳ˵���
void CfaceAnalysisDoc::OnUpdateColorBlack(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Color == ElementColor::BLACK);
}


void CfaceAnalysisDoc::OnUpdateColorRed(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Color == ElementColor::RED);
}


void CfaceAnalysisDoc::OnUpdateColorGreen(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Color == ElementColor::GREEN);
}


void CfaceAnalysisDoc::OnUpdateColorBlue(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Color == ElementColor::BLUE);
}


void CfaceAnalysisDoc::OnUpdateElementLine(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Element == ElementType::LINE);
}


void CfaceAnalysisDoc::OnUpdateElementRectangle(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Element == ElementType::RECTANGLE);
}


void CfaceAnalysisDoc::OnUpdateElementCircle(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Element == ElementType::CIRCLE);
}


void CfaceAnalysisDoc::OnUpdateElementCurve(CCmdUI *pCmdUI)
{
	// TODO:  �ڴ������������û����洦��������
	pCmdUI->SetCheck(m_Element == ElementType::CURVE);
}
