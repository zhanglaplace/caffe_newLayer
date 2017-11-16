
// faceAttributeDlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "faceAttribute.h"
#include "faceAttributeDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


DWORD WINAPI CaptureThread(LPVOID lpParameter){
	CfaceAttributeDlg* pDlg = (CfaceAttributeDlg*)lpParameter;
	Mat tmpImg,smallImg;
	float threshold[3] = { 0.7, 0.6, 0.6 };
	const int minSize = 40;
	const float factory = 0.709f;
	const int scale = 2;
	while (true){
		pDlg->m_capture >> tmpImg;
		if (!pDlg->m_isStart){ // ���Ҫ��ر�����ͷ
			break;
		}
		else{
			resize(tmpImg, smallImg, cvSize(tmpImg.cols / 2, tmpImg.rows / 2));
			vector<FaceInfos>faces = pDlg->m_fd->Detect(smallImg, minSize, threshold, factory, 3, pDlg->m_isLandmark);
			if (pDlg->m_isRecord){ // capture ׼��ע��
				pDlg->m_isRecord = false;
				if (faces.size() != 1){
					AfxMessageBox(_T("��ȷ����ǰֻ��һ���˳���������ͷ��Χ��"));
					pDlg->m_captureImg = tmpImg;
					pDlg->showImage(tmpImg, IDC_RECORD_PIC);
					Sleep(5);
				}
				else{
					int x = scale*(int) faces[0].bbox.xmin;
					int y = scale*(int)faces[0].bbox.ymin;
					int w = scale*(int)(faces[0].bbox.xmax - faces[0].bbox.xmin + 1);
					int h = scale*(int)(faces[0].bbox.ymax - faces[0].bbox.ymin + 1);
					pDlg->m_captureImg = tmpImg(Rect(x, y, w, h));
					pDlg->showImage(pDlg->m_captureImg, IDC_RECORD_PIC);
					Sleep(5);
				}
			}
			else{
				for (int i = 0; i < faces.size(); i++){
					int x = scale*(int)faces[i].bbox.xmin;
					int y = scale*(int)faces[i].bbox.ymin;
					int w = scale*(int)(faces[i].bbox.xmax - faces[i].bbox.xmin + 1);
					int h = scale*(int)(faces[i].bbox.ymax - faces[i].bbox.ymin + 1);
					cv::rectangle(tmpImg, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
				}
				pDlg->showImage(tmpImg, IDC_PICTURE);
			}
		}
	}
	return 0;
}


// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CfaceAttributeDlg �Ի���



CfaceAttributeDlg::CfaceAttributeDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CfaceAttributeDlg::IDD, pParent)
	, m_isStart(false)
	, m_isRecord(false)
	, m_isRecognition(false)
	, m_isLandmark(false)
	, m_resultName(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_fd = new FaceDetect();
	m_fr = new FaceRecognition();
}

void CfaceAttributeDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_RESUlT_NAME, m_resultName);
}

BEGIN_MESSAGE_MAP(CfaceAttributeDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_OPENCAMETA, &CfaceAttributeDlg::OnBnClickedBtnOpencameta)
	ON_BN_CLICKED(IDC_BTN_CLOSECAMETA, &CfaceAttributeDlg::OnBnClickedBtnClosecameta)
	ON_BN_CLICKED(IDC_BTN_CAPTURE, &CfaceAttributeDlg::OnBnClickedBtnCapture)
	ON_BN_CLICKED(IDC_REGISTER, &CfaceAttributeDlg::OnBnClickedRegister)
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_RECOGNITION, &CfaceAttributeDlg::OnBnClickedRecognition)
END_MESSAGE_MAP()


// CfaceAttributeDlg ��Ϣ�������

BOOL CfaceAttributeDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO:  �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void CfaceAttributeDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CfaceAttributeDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CfaceAttributeDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


/******************************************************
// ������:OnBnClickedBtnOpencameta
// ˵��: ������ͷ����������ͷ�߳�
// ����:�ŷ�
// ʱ��:
// ��ע:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedBtnOpencameta()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	if (!m_isStart){ // ����ͷδ������ʱ�����
		m_capture = VideoCapture(0);
		if (!m_capture.isOpened()){
			AfxMessageBox(_T("��ȷ������������ͷ������"));
		}
		else{
			m_isStart = true;
			HANDLE hThread = NULL;
			DWORD dwThreadID = 0;
			hThread = CreateThread(NULL, 0, CaptureThread, this, 0, &dwThreadID);
		}
	}
}


/******************************************************
// ������:showImage
// ˵��:��ʾͼƬ
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAttributeDlg::showImage(Mat& im, UINT ID){
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


/******************************************************
// ������:OnBnClickedBtnClosecameta
// ˵��: �ر�����ͷ
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedBtnClosecameta()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	m_isStart = false;
	Sleep(50);
	m_capture.release();
}

/******************************************************
// ������:OnBnClickedBtnCapture
// ˵��: ��ͼ�Դ�ע��
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedBtnCapture()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	m_isRecord = true;
	Sleep(50);
}


/******************************************************
// ������:OnBnClickedRegister
// ˵��:ע������
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedRegister()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	if (m_captureImg.empty()){
		MessageBox(_T("����ȷ��ע����ͼ���Ѿ���׽��", _T("��ʾ"), MB_OK));
		return ;
	}
	CRegisterBox box;
	CString personName;
	if (IDOK == box.DoModal()){
		personName = box.m_personName;
		char* strFileName = NULL;
#ifdef _UNICODE //���ַ�
		int nCharLen = WideCharToMultiByte(CP_ACP, 0, personName, -1, NULL, 0, NULL, NULL);
		strFileName = new char[nCharLen + 1];
		WideCharToMultiByte(CP_ACP, 0, personName, -1, strFileName, nCharLen + 1, NULL, NULL);
#else //�ǿ��ַ�
		strFileName = strImgFilename.GetBuffer(strImgFilename.GetLength() + 1);
		strImgFilename.ReleaseBuffer();
#endif
		int bCheck = m_fr->checkName(strFileName,strlen(strFileName));

		int recordNum = m_fr->getMaxRecordNum();
		m_fr->getLastLayerFeaturesFlip(m_captureImg, recordNum + 1);



		m_fr->saveFeaturesToFile(recordNum + 1, strFileName);
		m_fr->updateTotalFeature();
	}

}

/******************************************************
// ������:OnClose
// ˵��:�رմ���
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAttributeDlg::OnClose()
{
	// TODO:  �ڴ������Ϣ�����������/�����Ĭ��ֵ
	if (m_isStart){
		m_isStart = false;
		m_capture.release();
	}
	CDialogEx::OnClose();
}


void CfaceAttributeDlg::OnBnClickedRecognition()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������

	if (!m_isStart && m_captureImg.empty()){
		AfxMessageBox(_T("ȷ���Ƿ������ͷ"));
		return;
	}

	m_isRecognition = true;
	m_isRecord = true;
	Sleep(50);
	while (m_captureImg.empty()){ ; }

	m_fr->getLastLayerFeaturesFlip(m_captureImg, -1);
	int id = m_fr->compareTwoVectors();
	char resultName[50];
	if (id == -1){
		m_resultName = _T("�������ݿ�");
	}
	else{
		m_fr->getNameFromId(resultName, id);
		m_resultName = resultName;
	}

	UpdateData(false);
}
