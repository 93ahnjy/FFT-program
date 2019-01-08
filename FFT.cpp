#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#define SPECTRUM_WINDOW "spectrum magnitude"
#define WINDOW_NAME "Controller"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace cv;
using namespace std;

void fftshift(Mat& magI);
void fftpadd(Mat& I, Mat& padded);
void cvui_drawROI_scale(Mat& img_cpy, Mat& img_orig, int scale);

Point anchor;
Rect ROI;
vector<Rect> ROI_array;

int main()
{

	Mat I = imread("0_007.bmp", IMREAD_GRAYSCALE);
	Mat res_I;
	float thres = 0;
	int bs_thres = 0;
	int x = I.cols / 10;
	int y = I.rows / 10;

	Mat padded;                        // expand input image to optimal size
	fftpadd(I, padded);

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
										// dft결과는 2채널의 Mat로 나온다.

	dft(complexI, complexI);            // this way the result may fit in the source matrix



	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	split(complexI, planes);						// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);		// Re와 Im을 이용해 fft후 magnitude 저장.




	Mat magI = planes[0];
	

	magI += Scalar::all(1);                    // log scale로 변환.
	log(magI, magI);
	//magI = 20 * magI;
	fftshift(magI);
	



	// gui에 쓰일 '초기화용' 이미지들 저장.
	Mat complexI_orig;							
	complexI.copyTo(complexI_orig);

	Mat magI_orig;
	magI.copyTo(magI_orig);

	Mat magI_res;
	Mat magI_cpy;

	if (magI.cols > 10000 || magI.rows > 10000) {
		resize(magI, magI_res, Size(magI.cols / 10, magI.rows / 10));
		magI_res.copyTo(magI_cpy);
	}

	else {
		magI.copyTo(magI_res);
		magI.copyTo(magI_cpy);
	}




	// gui 밑바탕
	Mat frame(300,250, CV_8UC3);
	namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);
	cvui::watch(SPECTRUM_WINDOW);


	// bad sector 찾기
	Mat recon_I;
	Mat recon_thresd;
	vector<vector<cv::Point>> bs_contours;

	while (1) {


		

		complexI_orig.copyTo(complexI);											// fft 및 크기값 그래프 초기화
		magI_orig.copyTo(magI);
		


		frame = Scalar(49, 52, 49);
		cvui::printf(frame, 10, 10, 0.3, 0x00ff00, "Threshold");
		cvui::trackbar(frame, 10, 30, 200, &thres, (float)0., (float)1.);
		cvui::printf(frame, 10, 180, 0.3, 0x00ff00, "Bad sector threshold");
		cvui::trackbar(frame, 10, 200, 200, &bs_thres, (int)0., (int)255.);
		
		cvui::context(SPECTRUM_WINDOW);											// fft에서 threshold 및 Mask 설정
		magI_res.copyTo(magI_cpy);
		threshold(magI_cpy, magI_cpy, thres, 1, THRESH_TOZERO);	
		cvui_drawROI_scale(magI_cpy, magI, 10);
		cvui::imshow(SPECTRUM_WINDOW, magI_cpy);
		
		
		if (!recon_I.empty()) {
			threshold(recon_I, recon_thresd, bs_thres, 255, THRESH_BINARY);
			findContours(recon_thresd, bs_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			Mat findBadSect;
			cvtColor(res_I, findBadSect, COLOR_GRAY2RGB);
			drawContours(findBadSect, bs_contours, -1,  cv::Scalar(0, 0, 255), 2);
			imshow("bad sector", findBadSect);
		}
		



		cvui::context(WINDOW_NAME);												// 버튼 눌러야 아래의 idft 실행
		if (cvui::button(frame, 30, 120, "Delete all mask"))
			ROI_array.clear();
		if (!cvui::button(frame, 30, 80, "Start")) {
			cvui::imshow(WINDOW_NAME, frame);
			if (waitKey(20) == 27) {
				break;
			}
			continue;
		}
		cvui::update(WINDOW_NAME);

		
		threshold(magI, magI, thres, 1, THRESH_TOZERO);							// fftshift 시키면 다시 원래대로 돌리고 idft
		fftshift(magI);

		/*
		Mat temp;
		resize(magI, temp, Size(magI.cols / 10, magI.rows / 10));
		imshow("wowow", temp);
		waitKey(0);
		*/

		for (int i = 0; i < magI.rows; i++)
			for (int j = 0; j < magI.cols; j++){
				if (magI.at<float>(i,j) < thres){
					//cout << complexI.at<Vec2f>(i, j)[0] << endl;
					complexI.at<Vec2f>(i, j)[0] = 0;
					complexI.at<Vec2f>(i, j)[1] = 0;
					magI.at<float>(i, j) = 0;
					
				}
			}


		Mat inverseTransform;
		
		dft(complexI, inverseTransform, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);			// idft는 그냥 dft의 option으로 들어가 있다.
		normalize(inverseTransform, inverseTransform, 0, 1, NORM_MINMAX);

		Mat inv_outputs[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
		split(inverseTransform, inv_outputs);						// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		magnitude(inv_outputs[0], inv_outputs[1], inv_outputs[0]);		// Re와 Im을 이용해 fft후 magnitude 저장.
		
		recon_I = inv_outputs[0];


		if (I.cols > 10000 || I.rows > 10000) {													// 이미지가 너무 크면 크기 줄이자.
			resize(I, res_I, Size(x, y));
			resize(recon_I, recon_I, Size(x, y));
		}

		imshow("Input Image", res_I);    // Show the result
		imshow("Reconstructed", recon_I);
		recon_I.convertTo(recon_I, CV_8U, 255, 0);
		recon_I = recon_I - res_I;
		








		fftshift(magI);															// 위에서 fftshift가 풀렸으니 보여주기용으로 다시 적용.
		if (magI.cols > 10000 || I.rows > 10000)
			resize(magI, magI, Size(magI.cols / 10, magI.rows / 10));


		
		cvui::imshow(WINDOW_NAME, frame);


		if (waitKey(20) == 27) {
			break;
		}
	}
	return 0;
}


void fftshift(Mat& magI) {

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											 // viewable image form (float between values 0 and 1).	
}



void fftpadd(Mat& I, Mat& padded) {
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
}





void cvui_drawROI_scale(Mat& img_cpy, Mat& img_orig, int scale = 1) {


	// Select ROI

	if (cvui::mouse(cvui::DOWN)) {
		// Position the rectangle at the mouse pointer.
		anchor.x = cvui::mouse().x;
		anchor.y = cvui::mouse().y;
	}

	// Is any mouse button down (pressed)?
	if (cvui::mouse(cvui::IS_DOWN)) {

		//cvui::printf(img_cpy, anchor.x + 5, anchor.y + 5, 0.3, 0xff0000, "(%d,%d)", anchor.x, anchor.y);

		int mouse_x = cvui::mouse().x;
		int mouse_y = cvui::mouse().y;

		mouse_x = mouse_x < 0 ? 0 : mouse_x;
		mouse_y = mouse_y < 0 ? 0 : mouse_y;

		int width = mouse_x - anchor.x;
		int height = mouse_y - anchor.y;

		ROI.x = width < 0 ? anchor.x + width : anchor.x;
		ROI.y = height < 0 ? anchor.y + height : anchor.y;

		ROI.width = std::abs(width);
		ROI.height = std::abs(height);

		ROI.width = ROI.x + ROI.width > img_cpy.cols ? ROI.width + img_cpy.cols - (ROI.x + ROI.width) : ROI.width;
		ROI.height = ROI.y + ROI.height > img_cpy.rows ? ROI.height + img_cpy.rows - (ROI.y + ROI.height) : ROI.height;


		// Show the rectangle coordinates and size
		//cvui::printf(img_cpy, ROI.x + 5, ROI.y + 5, 0.3, 0x0000ff, "(%d,%d)", ROI.x, ROI.y);
		//cvui::printf(img_cpy, cvui::mouse().x + 5, cvui::mouse().y + 5, 0.3, 0xff0000, "w:%d, h:%d", ROI.width, ROI.height);
		cv::rectangle(img_cpy, ROI, Scalar(0));
	}

	if (cvui::mouse(cvui::UP))	
		ROI_array.push_back(ROI);


	for (int i = 0; i < ROI_array.size(); i++) {
		rectangle(img_cpy, ROI_array[i], Scalar(0), -1);

		Rect ROI_s(ROI_array[i].x*scale, ROI_array[i].y*scale, ROI_array[i].width*scale, ROI_array[i].height*scale);
		rectangle(img_orig, ROI_s, Scalar(0), -1);
	}
}

