
// faceAnalysisDlg.cpp : 实现文件
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
		if (pDlg->m_isClose)//退出循环
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



// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
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


// CfaceAnalysisDlg 对话框



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


// CfaceAnalysisDlg 消息处理程序

BOOL CfaceAnalysisDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
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

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO:  在此添加额外的初始化代码
	m_picHdc = GetDlgItem(IDC_PICTURE)->GetDC()->GetSafeHdc();
	GetDlgItem(IDC_PICTURE)->GetClientRect(&m_picRect);

	m_capHdc = GetDlgItem(IDC_CAP)->GetDC()->GetSafeHdc();
	GetDlgItem(IDC_CAP)->GetClientRect(&m_capRect);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
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

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CfaceAnalysisDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CfaceAnalysisDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


/******************************************************
// 函数名:OnBnClickedBtnCamera
// 说明: 打开摄像头，完成人脸检测
// 作者:张峰
// 时间:2017.11.04
// 备注:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedBtnCamera()
{
	// 摄像头人脸检测线程
	//namedWindow("capture");
	//namedWindow("face");
	//faceDetectThread->Start();
	if (!m_isStart){
		m_capture = VideoCapture(0);
		if (!m_capture.isOpened()){
			AfxMessageBox(_T("请确认至少有摄像头连上了"));
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
// 函数名:OnBnClickedCapture
// 说明:截取屏幕当前人脸图像进行注册
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedCapture()
{
	// TODO:  在此添加控件通知处理程序代码
	//EnterCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	//faceDetectThread->willRecord = true;
	//LeaveCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	m_isRecord = true;
	Sleep(100);
}



/******************************************************
// 函数名:OnBnClickedClosecamera
// 说明: 关闭摄像头
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedClosecamera()
{
	// TODO:  在此添加控件通知处理程序代码
	//faceDetectThread->Stop();
	//destroyAllWindows();
	m_isStart = false;
	m_isClose = true;
	m_capture.release();
}

/******************************************************
// 函数名:OnBnClickedBtnRegesiter
// 说明: 人物在线注册
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedBtnRegesiter()
{
	// TODO:  在此添加控件通知处理程序代码
	faceImgShow = cv::Mat(faceImage);
	if (!faceDetectThread->isValidImage(faceImgShow)) return;
	CRegisterBox box;
	CString personName;
	if (IDOK == box.DoModal()){ // 点击确定,记录当前的姓名
		personName = box.m_personName;
		imwrite("trst.bmp", faceImgShow);
		int recordNum = frObj->getMaxRecordNum();
		frObj->getLastLayerFeaturesFlip(faceImgShow, recordNum + 1);
		


		char* strFileName = NULL;
#ifdef _UNICODE //宽字符
		int nCharLen =  WideCharToMultiByte(CP_ACP, 0, personName, -1, NULL, 0, NULL, NULL);
		strFileName = new char[nCharLen + 1];
		WideCharToMultiByte(CP_ACP, 0, personName, -1, strFileName, nCharLen + 1, NULL, NULL);
#else //非宽字符
		strFileName = strImgFilename.GetBuffer(strImgFilename.GetLength() + 1);
		strImgFilename.ReleaseBuffer();
#endif
		
		frObj->saveFeaturesToFile(recordNum + 1, strFileName);
		//frObj->loadTotalFacesFromFile("vsFaceDb.txt", frObj->registeredFaces); //所有特征向量都加在到了registeredFaces中了
		frObj->UpdateTotalFeature();

		
	}

}



/******************************************************
// 函数名:OnBnClickedBtnRecognition
// 说明:人脸识别匹配
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedBtnRecognition()
{
	// TODO:  在此添加控件通知处理程序代码
	EnterCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	faceDetectThread->willRecord = true;
	LeaveCriticalSection(&(faceDetectThread->m_CriticalSectionForFaceImg));
	faceImgShow = cv::Mat(faceImage);
	if (!faceDetectThread->isValidImage(faceImgShow)) return;
	frObj->getLastLayerFeaturesFlip(faceImgShow, 1); // 保存在feature中了
	int id = frObj->compareTwoVectors();
	char resultName[50];
	frObj->getNameFromId(resultName, id);
	m_name = resultName;
	UpdateData(false);
}

/******************************************************
// 函数名:OnBnClickedCancel
// 说明:
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
void CfaceAnalysisDlg::OnBnClickedCancel()
{
	// TODO:  在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
}



/******************************************************
// 函数名:showImage
// 说明:显示图片
// 作者:张峰
// 时间:2017.11.15
// 备注:
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