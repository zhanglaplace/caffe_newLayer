// RegisterBox.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "faceAnalysis.h"
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
	ON_BN_CLICKED(ID_OK, &CRegisterBox::OnBnClickedOk)
	ON_BN_CLICKED(ID_CANCEL, &CRegisterBox::OnBnClickedCancel)
END_MESSAGE_MAP()


// CRegisterBox ��Ϣ�������


void CRegisterBox::OnBnClickedOk()
{
	UpdateData(TRUE);
	CDialog::OnOK();
}


void CRegisterBox::OnBnClickedCancel()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CDialog::OnCancel();
}
