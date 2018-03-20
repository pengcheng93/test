//
//  main.cpp
//  test
//
//  Created by visgroup on 2018/3/16.
//  Copyright © 2018年 visgroup. All rights reserved.
//

#include <iostream>
#include "facealignment.h"
#include "ldmarkmodel.h"

int main(int argc, const char * argv[]) {
    // insert code here...
    ldmarkmodel model;
    load_ldmarkmodel("./res/landmarkmodel.bin", model);
    
    cv::CascadeClassifier cascade;
    cascade.load("./res/haarcascade_frontalface_alt.xml");
    
    cv::Mat image=cv::imread("1.jpg");
    cv::Rect face;
    cv::Mat shape=predictfaceshape(image, model, cascade,face);
    cout<<shape<<std::endl;
    cout<<face<<std::endl;
    
    return 0;
}
