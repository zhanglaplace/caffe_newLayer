
// faceAttributeDlg.cpp : 实现文件
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
		if (!pDlg->m_isStart){ // 如果要求关闭摄像头
			break;
		}
		else{
			resize(tmpImg, smallImg, cvSize(tmpImg.cols / 2, tmpImg.rows / 2));
			vector<FaceInfos>faces = pDlg->m_fd->Detect(smallImg, minSize, threshold, factory, 3, pDlg->m_isLandmark);
			if (pDlg->m_isRecord){ // capture 准备注册
				pDlg->m_isRecord = false;
				if (faces.size() != 1){
					AfxMessageBox(_T("请确保当前只有一个人出现在摄像头范围内"));
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


// CfaceAttributeDlg 对话框



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


// CfaceAttributeDlg 消息处理程序

BOOL CfaceAttributeDlg::OnInitDialog()
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

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
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

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CfaceAttributeDlg::OnPaint()
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
HCURSOR CfaceAttributeDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


/******************************************************
// 函数名:OnBnClickedBtnOpencameta
// 说明: 打开摄像头，开启摄像头线程
// 作者:张峰
// 时间:
// 备注:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedBtnOpencameta()
{
	// TODO:  在此添加控件通知处理程序代码
	if (!m_isStart){ // 摄像头未启动的时候才做
		m_capture = VideoCapture(0);
		if (!m_capture.isOpened()){
			AfxMessageBox(_T("请确认至少有摄像头连上了"));
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
// 函数名:showImage
// 说明:显示图片
// 作者:张峰
// 时间:2017.11.15
// 备注:
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
// 函数名:OnBnClickedBtnClosecameta
// 说明: 关闭摄像头
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedBtnClosecameta()
{
	// TODO:  在此添加控件通知处理程序代码
	m_isStart = false;
	Sleep(50);
	m_capture.release();
}

/******************************************************
// 函数名:OnBnClickedBtnCapture
// 说明: 截图以待注册
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedBtnCapture()
{
	// TODO:  在此添加控件通知处理程序代码
	m_isRecord = true;
	Sleep(50);
}


/******************************************************
// 函数名:OnBnClickedRegister
// 说明:注册新人
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAttributeDlg::OnBnClickedRegister()
{
	// TODO:  在此添加控件通知处理程序代码
	if (m_captureImg.empty()){
		MessageBox(_T("请先确保注册人图像已经捕捉到", _T("提示"), MB_OK));
		return ;
	}
	CRegisterBox box;
	CString personName;
	if (IDOK == box.DoModal()){
		personName = box.m_personName;
		char* strFileName = NULL;
#ifdef _UNICODE //宽字符
		int nCharLen = WideCharToMultiByte(CP_ACP, 0, personName, -1, NULL, 0, NULL, NULL);
		strFileName = new char[nCharLen + 1];
		WideCharToMultiByte(CP_ACP, 0, personName, -1, strFileName, nCharLen + 1, NULL, NULL);
#else //非宽字符
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
// 函数名:OnClose
// 说明:关闭窗体
// 作者:张峰
// 时间:2017.11.15
// 备注:
/*******************************************************/
void CfaceAttributeDlg::OnClose()
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值
	if (m_isStart){
		m_isStart = false;
		m_capture.release();
	}
	CDialogEx::OnClose();
}


void CfaceAttributeDlg::OnBnClickedRecognition()
{
	// TODO:  在此添加控件通知处理程序代码

	if (!m_isStart && m_captureImg.empty()){
		AfxMessageBox(_T("确认是否打开摄像头"));
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
		m_resultName = _T("不在数据库");
	}
	else{
		m_fr->getNameFromId(resultName, id);
		m_resultName = resultName;
	}

	UpdateData(false);
}
