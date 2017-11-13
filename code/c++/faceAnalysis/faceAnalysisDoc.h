
// faceAnalysisDoc.h : CfaceAnalysisDoc 类的接口
//


#pragma once

#include "ElementColor.h"
#include "ElementType.h"

class CfaceAnalysisDoc : public CDocument
{
protected: // 仅从序列化创建
	CfaceAnalysisDoc();
	DECLARE_DYNCREATE(CfaceAnalysisDoc)

// 特性
public:

// 操作
public:

// 重写
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
#ifdef SHARED_HANDLERS
	virtual void InitializeSearchContent();
	virtual void OnDrawThumbnail(CDC& dc, LPRECT lprcBounds);
#endif // SHARED_HANDLERS

// 实现
public:
	virtual ~CfaceAnalysisDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 生成的消息映射函数
protected:
	DECLARE_MESSAGE_MAP()

#ifdef SHARED_HANDLERS
	// 用于为搜索处理程序设置搜索内容的 Helper 函数
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
