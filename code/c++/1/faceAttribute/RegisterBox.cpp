// RegisterBox.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "faceAttribute.h"
#include "RegisterBox.h"
#include "afxdialogex.h"


// CRegisterBox �Ի���

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
	ON_BN_CLICKED(IDOK, &CRegisterBox::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &CRegisterBox::OnBnClickedCancel)
END_MESSAGE_MAP()


// CRegisterBox ��Ϣ�������


void CRegisterBox::OnBnClickedOk()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	UpdateData(TRUE);
	CDialog::OnOK();
}


void CRegisterBox::OnBnClickedCancel()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CDialog::OnCancel();
}
