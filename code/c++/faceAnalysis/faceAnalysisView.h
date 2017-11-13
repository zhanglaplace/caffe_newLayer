
// faceAnalysisView.h : CfaceAnalysisView ��Ľӿ�
//

#pragma once


class CfaceAnalysisView : public CView
{
protected: // �������л�����
	CfaceAnalysisView();
	DECLARE_DYNCREATE(CfaceAnalysisView)

// ����
public:
	CfaceAnalysisDoc* GetDocument() const;

// ����
public:

// ��д
public:
	virtual void OnDraw(CDC* pDC);  // ��д�Ի��Ƹ���ͼ
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// ʵ��
public:
	virtual ~CfaceAnalysisView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// ���ɵ���Ϣӳ�亯��
protected:
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // faceAnalysisView.cpp �еĵ��԰汾
inline CfaceAnalysisDoc* CfaceAnalysisView::GetDocument() const
   { return reinterpret_cast<CfaceAnalysisDoc*>(m_pDocument); }
#endif

