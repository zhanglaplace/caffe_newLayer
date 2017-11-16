
// faceAnalysisDlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "faceAnalysis.h"
#include "faceAnalysisDlg.h"
#include "afxdialogex.h"
#include "processThread.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


DWORD WINAPI CaptureThread(LPVOID lpParameter)
{
	CfaceAnalysisDlg* pDlg = (CfaceAnalysisDlg*)lpParameter;
	Mat tmpImg;
	while (true)
	{
		pDlg->m_capture >> tmpImg;
		if (pDlg->m_isClose)//�˳�ѭ��
			break;
		if (pDlg->m_isRecord)
		{
			pDlg->m_isRecord = false;
			pDlg->captureImg = tmpImg;
			pDlg->showImage(tmpImg, IDC_CAP);
			Sleep(5);
		}
		pDlg->showImage(tmpImg, IDC_PICTURE);
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


// CfaceAnalysisDlg �Ի���



CfaceAnalysisDlg::CfaceAnalysisDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CfaceAnalysisDlg::IDD, pParent)
	, m_name(_T(""))
	, m_isStart(false)
	, m_isRecord(false)
	, m_isRecognition(false)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	faceImage = cvCreateImage(cvSize(96,112),8,3);
	//faceDetectThread = new ProcessThread(1, this, faceImage);
	frObj = new FaceRecognition();
}

void CfaceAnalysisDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_RESULT, m_name);
}

BEGIN_MESSAGE_MAP(CfaceAnalysisDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_CAMERA, &CfaceAnalysisDlg::OnBnClickedBtnCamera)
	ON_BN_CLICKED(IDC_BTN_CAPTURE, &CfaceAnalysisDlg::OnBnClickedCapture)
	ON_BN_CLICKED(IDC_BTN_CLOSECAMERA, &CfaceAnalysisDlg::OnBnClickedClosecamera)
	ON_BN_CLICKED(IDC_BTN_REGESITER, &CfaceAnalysisDlg::OnBnClickedBtnRegesiter)
	ON_BN_CLICKED(IDC_BTN_RECOGNITION, &CfaceAnalysisDlg::OnBnClickedBtnRecognition)
	ON_BN_CLICKED(IDCANCEL, &CfaceAnalysisDlg::OnBnClickedCancel)
END_MESSAGE_MAP()


// CfaceAnalysisDlg ��Ϣ�������

BOOL CfaceAnalysisDlg::OnInitDialog()
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
	m_picHdc = GetDlgItem(IDC_PICTURE)->GetDC()->GetSafeHdc();
	GetDlgItem(IDC_PICTURE)->GetClientRect(&m_picRect);

	m_capHdc = GetDlgItem(IDC_CAP)->GetDC()->GetSafeHdc();
	GetDlgItem(IDC_CAP)->GetClientRect(&m_capRect);

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void CfaceAnalysisDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CfaceAnalysisDlg::OnPaint()
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
HCURSOR CfaceAnalysisDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


/******************************************************
// ������:OnBnClickedBtnCamera
// ˵��: ������ͷ������������
// ����:�ŷ�
// ʱ��:2017.11.04
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedBtnCamera()
{
	// ����ͷ��������߳�
	//namedWindow("capture");
	//namedWindow("face");
	//faceDetectThread->Start();
	if (!m_isStart){
		m_capture = VideoCapture(0);
		if (!m_capture.isOpened()){
			AfxMessageBox(_T("��ȷ������������ͷ������"));
		}
		else{
			m_isStart = true;
			m_isClose = false;
			HANDLE hThread = NULL;
			DWORD dwThreadID = 0;
			hThread = CreateThread(NULL, 0, CaptureThread, this, 0, &dwThreadID);
		}
	}
}

/******************************************************
// ������:OnBnClickedCapture
// ˵��:��ȡ��Ļ��ǰ����ͼ�����ע��
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedCapture()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	//EnterCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	//faceDetectThread->willRecord = true;
	//LeaveCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	m_isRecord = true;
	Sleep(100);
}



/******************************************************
// ������:OnBnClickedClosecamera
// ˵��: �ر�����ͷ
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedClosecamera()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	//faceDetectThread->Stop();
	//destroyAllWindows();
	m_isStart = false;
	m_isClose = true;
	m_capture.release();
}

/******************************************************
// ������:OnBnClickedBtnRegesiter
// ˵��: ��������ע��
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedBtnRegesiter()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	faceImgShow = cv::Mat(faceImage);
	if (!faceDetectThread->isValidImage(faceImgShow)) return;
	CRegisterBox box;
	CString personName;
	if (IDOK == box.DoModal()){ // ���ȷ��,��¼��ǰ������
		personName = box.m_personName;
		imwrite("trst.bmp", faceImgShow);
		int recordNum = frObj->getMaxRecordNum();
		frObj->getLastLayerFeaturesFlip(faceImgShow, recordNum + 1);
		


		char* strFileName = NULL;
#ifdef _UNICODE //���ַ�
		int nCharLen =  WideCharToMultiByte(CP_ACP, 0, personName, -1, NULL, 0, NULL, NULL);
		strFileName = new char[nCharLen + 1];
		WideCharToMultiByte(CP_ACP, 0, personName, -1, strFileName, nCharLen + 1, NULL, NULL);
#else //�ǿ��ַ�
		strFileName = strImgFilename.GetBuffer(strImgFilename.GetLength() + 1);
		strImgFilename.ReleaseBuffer();
#endif
		
		frObj->saveFeaturesToFile(recordNum + 1, strFileName);
		//frObj->loadTotalFacesFromFile("vsFaceDb.txt", frObj->registeredFaces); //�����������������ڵ���registeredFaces����
		frObj->UpdateTotalFeature();

		
	}

}



/******************************************************
// ������:OnBnClickedBtnRecognition
// ˵��:����ʶ��ƥ��
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedBtnRecognition()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	EnterCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	faceDetectThread->willRecord = true;
	LeaveCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	faceImgShow = cv::Mat(faceImage);
	if (!faceDetectThread->isValidImage(faceImgShow)) return;
	frObj->getLastLayerFeaturesFlip(faceImgShow, 1); // ������feature����
	int id = frObj->compareTwoVectors();
	char resultName[50];
	frObj->getNameFromId(resultName, id);
	m_name = resultName;
	UpdateData(false);
}

/******************************************************
// ������:OnBnClickedCancel
// ˵��:
// ����:�ŷ�
// ʱ��:
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedCancel()
{
	// TODO:  �ڴ���ӿؼ�֪ͨ����������
	CDialogEx::OnCancel();
}



/******************************************************
// ������:showImage
// ˵��:��ʾͼƬ
// ����:�ŷ�
// ʱ��:2017.11.15
// ��ע:
/*******************************************************/
void CfaceAnalysisDlg::showImage(Mat& im, UINT ID){
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