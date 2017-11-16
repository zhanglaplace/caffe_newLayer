// RegisterBox.cpp : 实现文件
//

#include "stdafx.h"
#include "faceAnalysis.h"
#include "RegisterBox.h"
#include "afxdialogex.h"


// CRegisterBox 对话框

IMPLEMENT_DYNAMIC(CRegisterBox, CDialog)

CRegisterBox::CRegisterBox(CWnd* pParent /*=NULL*/)
	: CDialog(CRegisterBox::IDD, pParent)
	, m_personName(_T(""))
{

}

CRegisterBox::~CRegisterBox()
{
}

void CRegisterBox::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_NAME, m_personName);
}


BEGIN_MESSAGE_MAP(CRegisterBox, CDialog)
	ON_BN_CLICKED(ID_OK, &CRegisterBox::OnBnClickedOk)
	ON_BN_CLICKED(ID_CANCEL, &CRegisterBox::OnBnClickedCancel)
END_MESSAGE_MAP()


// CRegisterBox 消息处理程序


void CRegisterBox::OnBnClickedOk()
{
	UpdateData(TRUE);
	CDialog::OnOK();
}


void CRegisterBox::OnBnClickedCancel()
{
	// TODO:  在此添加控件通知处理程序代码
	CDialog::OnCancel();
}
