
// faceAnalysisView.cpp : CfaceAnalysisView 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
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
	// 标准打印命令
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
END_MESSAGE_MAP()

// CfaceAnalysisView 构造/析构

CfaceAnalysisView::CfaceAnalysisView()
{
	// TODO:  在此处添加构造代码

}

CfaceAnalysisView::~CfaceAnalysisView()
{
}

BOOL CfaceAnalysisView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO:  在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式

	return CView::PreCreateWindow(cs);
}

// CfaceAnalysisView 绘制

void CfaceAnalysisView::OnDraw(CDC* /*pDC*/)
{
	CfaceAnalysisDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO:  在此处为本机数据添加绘制代码
}


// CfaceAnalysisView 打印

BOOL CfaceAnalysisView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 默认准备
	return DoPreparePrinting(pInfo);
}

void CfaceAnalysisView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO:  添加额外的打印前进行的初始化过程
}

void CfaceAnalysisView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO:  添加打印后进行的清理过程
}


// CfaceAnalysisView 诊断

#ifdef _DEBUG
void CfaceAnalysisView::AssertValid() const
{
	CView::AssertValid();
}

void CfaceAnalysisView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CfaceAnalysisDoc* CfaceAnalysisView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CfaceAnalysisDoc)));
	return (CfaceAnalysisDoc*)m_pDocument;
}
#endif //_DEBUG


// CfaceAnalysisView 消息处理程序
