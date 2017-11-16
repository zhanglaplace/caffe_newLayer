// Compare.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "faceAttribute.h"
#include "Compare.h"
#include "afxdialogex.h"
#include "faceAttributeDlg.h"
#include "faceRecognition.h"
#include "faceDetect.h"

// CCompare �Ի���

IMPLEMENT_DYNAMIC(CCompare, CDialogEx)

CCompare::CCompare(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCompare::IDD, pParent)
	, m_similarity(_T(""))
	, m_file1(_T(""))
{

}

CCompare::~CCompare()
{
}

void CCompare::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_SIMILARITY, m_similarity);
}


BEGIN_MESSAGE_MAP(CCompare, CDialogEx)
	ON_BN_CLICKED(IDC_BTN_1, &CCompare::OnBnClickedBtn1)
	ON_BN_CLICKED(IDC_BTN_2, &CCompare::OnBnClickedBtn2)
	ON_BN_CLICKED(IDC_BTN_CAMPARE, &CCompare::OnBnClickedBtnCampare)
END_MESSAGE_MAP()


// CCompare ��Ϣ�������


void CCompare::OnBnClickedBtn1()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CFileDialog fileDlg(TRUE,  // TRUE��Open��FALSE����Save As�ļ��Ի���
		_T(".jpg"),  // Ĭ�ϵĴ��ļ�������
		_T(""), // Ĭ�ϴ򿪵��ļ��� 
		OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR,  // ��ѡ��
		_T("jpg�ļ�(*.jpg)|*.jpg|�����ļ�(*.*) |*.*||")  // �򿪵��ļ�����
		);

	//fileDlg.m_ofn.lpstrInitialDir = strPath;//��ʼ��·����
	if (fileDlg.DoModal() == IDOK){
		m_file1 = fileDlg.GetPathName();//����ѡ���������ļ����ƣ�
		char* strFileName = NULL;
#ifdef _UNICODE //���ַ�
		int nCharLen = WideCharToMultiByte(CP_ACP, 0, m_file1, -1, NULL, 0, NULL, NULL);
		strFileName = new char[nCharLen + 1];
		WideCharToMultiByte(CP_ACP, 0, m_file1, -1, strFileName, nCharLen + 1, NULL, NULL);
#else //�ǿ��ַ�
		strFileName = strImgFilename.GetBuffer(strImgFilename.GetLength() + 1);
		strImgFilename.ReleaseBuffer();
#endif
		m_img1 = imread(strFileName);
		showImage(m_img1, IDC_PICTURE_1);
	}
}


void CCompare::OnBnClickedBtn2()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CFileDialog fileDlg(TRUE,  // TRUE��Open��FALSE����Save As�ļ��Ի���
		_T(".jpg"),  // Ĭ�ϵĴ��ļ�������
		_T(""), // Ĭ�ϴ򿪵��ļ��� 
		OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR,  // ��ѡ��
		_T("jpg�ļ�(*.jpg)|*.jpg|�����ļ�(*.*) |*.*||")  // �򿪵��ļ�����
		);
	//fileDlg.m_ofn.lpstrInitialDir = strPath;//��ʼ��·����
	if (fileDlg.DoModal() == IDOK){
		m_file2 = fileDlg.GetPathName();//����ѡ���������ļ����ƣ�
		char* strFileName = NULL;
#ifdef _UNICODE //���ַ�
		int nCharLen = WideCharToMultiByte(CP_ACP, 0, m_file2, -1, NULL, 0, NULL, NULL);
		strFileName = new char[nCharLen + 1];
		WideCharToMultiByte(CP_ACP, 0, m_file2, -1, strFileName, nCharLen + 1, NULL, NULL);
#else //�ǿ��ַ�
		strFileName = strImgFilename.GetBuffer(strImgFilename.GetLength() + 1);
		strImgFilename.ReleaseBuffer();
#endif
		m_img2 = imread(strFileName);
		showImage(m_img2, IDC_PICTURE_2);
	}
}


/******************************************************
// ������:showImage
// ˵��:��ʾͼƬ
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CCompare::showImage(Mat& im, UINT ID){
	if (im.empty()){
		return;
	}
	CRect rect;
	GetDlgItem(ID)->GetClientRect(&rect);
	HDC hDC = GetDlgItem(ID)->GetDC()->GetSafeHdc();
	CvvImage cimg;
	IplImage cpy = im;
	cimg.CopyOf(&cpy);
	cimg.DrawToHDC(hDC, &rect);
}

void CCompare::OnBnClickedBtnCampare()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CfaceAttributeDlg* pDlgface = (CfaceAttributeDlg*)(this->GetParent());
	float threshold[3] = { 0.7, 0.6, 0.6 };
	const int minSize = 40;
	const float factory = 0.709f;
	const int scale = 2;
	Mat  roi1,roi2;
	vector<FaceInfos>faces1 = pDlgface->m_fd->Detect(m_img1, minSize, threshold, factory, 3, pDlgface->m_isLandmark);
	vector<FaceInfos>faces2 = pDlgface->m_fd->Detect(m_img2, minSize, threshold, factory, 3, pDlgface->m_isLandmark);
	if (faces1.size() == 0)
	{
		roi1  = m_img1;
	}
	else{
		int x = (int)faces1[0].bbox.xmin;
		int y = (int)faces1[0].bbox.ymin;
		int w = (int)(faces1[0].bbox.xmax - faces1[0].bbox.xmin + 1);
		int h = (int)(faces1[0].bbox.ymax - faces1[0].bbox.ymin + 1);
		roi1 = m_img1(Rect(x, y, w, h));
	}
	if (faces2.size() == 0)
	{
		roi2 = m_img2;
	}
	else{
		int x1 = (int)faces2[0].bbox.xmin;
		int y1 = (int)faces2[0].bbox.ymin;
		int w1 = (int)(faces2[0].bbox.xmax - faces2[0].bbox.xmin + 1);
		int h1 = (int)(faces2[0].bbox.ymax - faces2[0].bbox.ymin + 1);
		roi2 = m_img2(Rect(x1, y1, w1, h1));
	}

	vector<float>vec1 = pDlgface->m_fr->getLastLayerFeaturesFlip(roi1);
	vector<float>vec2 = pDlgface->m_fr->getLastLayerFeaturesFlip(roi2);

	float t1 = 0.f, t2 = 0.f, t3 = 0.f;
	for (int i = 0; i < vec1.size(); i++){
		t1 += vec1[i] * vec2[i];
		t2 += vec1[i] * vec1[i];
		t3 += vec2[i] * vec2[i];
	}
	float similarity = t1 / (sqrt(t2)*sqrt(t3));
	char result[50];
	sprintf(result, "%f", similarity/2+0.5);
	m_similarity = result;
	UpdateData(false);

}
