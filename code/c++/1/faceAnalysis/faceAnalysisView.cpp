
// faceAnalysisView.cpp : CfaceAnalysisView ���ʵ��
//

#include "stdafx.h"
// SHARED_HANDLERS ������ʵ��Ԥ��������ͼ������ɸѡ�������
// ATL ��Ŀ�н��ж��壬�����������Ŀ�����ĵ����롣
#ifndef SHARED_HANDLERS
#include "faceAnalysis.h"
#endif

#include "faceAnalysisDoc.h"
#include "faceAnalysisView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CfaceAnalysisView

IMPLEMENT_DYNCREATE(CfaceAnalysisView, CView)

BEGIN_MESSAGE_MAP(CfaceAnalysisView, CView)
	// ��׼��ӡ����
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
END_MESSAGE_MAP()

// CfaceAnalysisView ����/����

CfaceAnalysisView::CfaceAnalysisView()
{
	// TODO:  �ڴ˴���ӹ������

}

CfaceAnalysisView::~CfaceAnalysisView()
{
}

BOOL CfaceAnalysisView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO:  �ڴ˴�ͨ���޸�
	//  CREATESTRUCT cs ���޸Ĵ��������ʽ

	return CView::PreCreateWindow(cs);
}

// CfaceAnalysisView ����

void CfaceAnalysisView::OnDraw(CDC* /*pDC*/)
{
	CfaceAnalysisDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO:  �ڴ˴�Ϊ����������ӻ��ƴ���
}


// CfaceAnalysisView ��ӡ

BOOL CfaceAnalysisView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// Ĭ��׼��
	return DoPreparePrinting(pInfo);
}

void CfaceAnalysisView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO:  ��Ӷ���Ĵ�ӡǰ���еĳ�ʼ������
}

void CfaceAnalysisView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO:  ��Ӵ�ӡ����е��������
}


// CfaceAnalysisView ���

#ifdef _DEBUG
void CfaceAnalysisView::AssertValid() const
{
	CView::AssertValid();
}

void CfaceAnalysisView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CfaceAnalysisDoc* CfaceAnalysisView::GetDocument() const // �ǵ��԰汾��������
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CfaceAnalysisDoc)));
	return (CfaceAnalysisDoc*)m_pDocument;
}
#endif //_DEBUG


// CfaceAnalysisView ��Ϣ�������
