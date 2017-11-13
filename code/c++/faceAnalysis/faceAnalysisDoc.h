
// faceAnalysisDoc.h : CfaceAnalysisDoc ��Ľӿ�
//


#pragma once

#include "ElementColor.h"
#include "ElementType.h"

class CfaceAnalysisDoc : public CDocument
{
protected: // �������л�����
	CfaceAnalysisDoc();
	DECLARE_DYNCREATE(CfaceAnalysisDoc)

// ����
public:

// ����
public:

// ��д
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
#ifdef SHARED_HANDLERS
	virtual void InitializeSearchContent();
	virtual void OnDrawThumbnail(CDC& dc, LPRECT lprcBounds);
#endif // SHARED_HANDLERS

// ʵ��
public:
	virtual ~CfaceAnalysisDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// ���ɵ���Ϣӳ�亯��
protected:
	DECLARE_MESSAGE_MAP()

#ifdef SHARED_HANDLERS
	// ����Ϊ����������������������ݵ� Helper ����
	void SetSearchContent(const CString& value);
#endif // SHARED_HANDLERS
public:
	afx_msg void OnColorBlack();
	afx_msg void OnColorRed();
	afx_msg void OnColorGreen();
	afx_msg void OnColorBlue();
	afx_msg void OnElementLine();
	afx_msg void OnElementRectangle();
	afx_msg void OnElementCircle();
	afx_msg void OnElementCurve();
public:
	ElementType GetElementType()const{ return m_Element; }
	ElementColor GetElementColor() const{ return m_Color; }
protected:
	ElementType m_Element;
	ElementColor m_Color;
};
