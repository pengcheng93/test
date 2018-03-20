#pragma once
#ifndef UTILS_H_
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <cstdarg>
#include <opencv2/opencv.hpp>
#pragma warning(disable:4996)
#define landmark_num 68
using namespace std;

void facedetect(cv::Mat img, vector<cv::Rect> &bboxes);
cv::Rect getBBox(cv::Mat &img, cv::Mat_<double> &shape);
void geteyecenter(cv::Mat_<double> shape, cv::Point2f &leftcenter, cv::Point2f &rightcenter);
cv::Rect perturb(cv::Rect facebox);
cv::Mat align_mean(cv::Mat mean, cv::Rect facebox);
cv::Rect_<int> get_enclosing_bbox(cv::Mat landmarks);
cv::Mat drawShapeInImage(const cv::Mat &img, const cv::Mat &shape, const cv::Rect &bbox);
bool rectInImage(cv::Rect rect, cv::Mat image);
bool inMat(cv::Point p, int rows, int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
cv::Rect geteyerect(cv::Mat shape, int minlandmarkidx, int maxlandmarkidx);
double calDistanceDiff(std::vector<cv::Point2f> curPoints, std::vector<cv::Point2f> lastPoints);

void facedetectopencv(cv::Mat img, vector<cv::Rect> &bboxes, cv::CascadeClassifier& cascade);
#endif // !UTILS_H_
