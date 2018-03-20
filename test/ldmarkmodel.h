#pragma once
#ifndef LDMARKMODEL_H_
#define LDMARKMODEL_H_

#include <iostream>
#include <vector>
#include <fstream>
#include "feature_descriptor.h"
#include "utils.h"

//回归器类
class LinearRegressor{

public:
	LinearRegressor();

	bool learn(cv::Mat &data, cv::Mat &labels);

	double test(cv::Mat data, cv::Mat labels);

	cv::Mat predict(cv::Mat values);

	void convert(std::vector<int> &tar_LandmarkIndex);
private:
	cv::Mat weights;
	//cv::Mat eigenvectors;
	cv::Mat meanvalue;
	cv::Mat x;
	bool isPCA;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(weights, meanvalue, x);
	}
};


class ldmarkmodel{

public:
	ldmarkmodel();

	ldmarkmodel(std::vector<std::vector<int>> LandmarkIndexs, std::vector<int> eyes_index, cv::Mat meanShape, std::vector<HoGParam> HoGParams, std::vector<LinearRegressor> LinearRegressors);

	cv::Mat predict(const cv::Mat& src);

    cv::Mat predictbyOpencv(const cv::Mat src,cv::CascadeClassifier cascade,cv::Rect &FaceRect);

	int multidetect(const cv::Mat& src);


	int  track(const cv::Mat& src, cv::Mat& current_shape, bool isDetFace = false);
	int trackopencv(const cv::Mat& src, cv::Mat& current_shape, bool isDetFace, cv::CascadeClassifier cascade);

	void printmodel();

	void convert(std::vector<int> &full_eyes_Indexs);

private:
	cv::Rect faceBox;


	std::vector<std::vector<int>> LandmarkIndexs;
	std::vector<int> eyes_index;
	cv::Mat meanShape;
	std::vector<HoGParam> HoGParams;
	bool isNormal;
	std::vector<LinearRegressor> LinearRegressors;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(LandmarkIndexs, eyes_index, meanShape, HoGParams, isNormal, LinearRegressors);
	}

	//add
	cv::Mat oldshape;
};

//加载模型
bool load_ldmarkmodel(std::string filename, ldmarkmodel &model);

//保存模型
void save_ldmarkmodel(ldmarkmodel model, std::string filename);


#endif
