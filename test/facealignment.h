//
//  facealignment.h
//  FaceAlignmentLib
//
//  Created by visgroup on 2018/3/16.
//  Copyright © 2018年 visgroup. All rights reserved.
//

#ifndef facealignment_h
#define facealignment_h
#include "ldmarkmodel.h"

cv::Mat predictfaceshape(cv::Mat image,ldmarkmodel model,cv::CascadeClassifier cascade,cv::Rect &face);


#endif /* facealignment_h */
