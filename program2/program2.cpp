#include <iostream>
#include "opencv2/opencv.hpp"

#define NUM_COMNMAND_LINE_ARGUMENTS 1
static std::map<std::string, int> change({
    {"Penny", 0},
    {"Nickel", 0},
    {"Dime", 0},
    {"Quarter", 0},
});

double value(cv::Size size){
    double approxDiam = (size.width + size.height) / 2;
    if(approxDiam > 310 && approxDiam < 320){
        change["Penny"]++;
        return 0.01;
    } else if(approxDiam > 345 && approxDiam < 370){
        change["Nickel"]++;
        return 0.05;
    } else if(approxDiam > 290 && approxDiam < 300){
        change["Dime"]++;
        return 0.10;
    } else if(approxDiam > 390 && approxDiam < 415){
        change["Quarter"]++;
        return 0.25;
    }
    return 0.0;
} 

int main(int argc, char **argv){
    std::string inputFileName;
    cv::Mat imageGray, imageIn;

    if(argc != NUM_COMNMAND_LINE_ARGUMENTS + 1){
        std::printf("USAGE: %s <file_path> \n", argv[0]);
    } else{
        inputFileName = argv[1];
    }

    imageIn = cv::imread(inputFileName, cv::IMREAD_COLOR);
    if(!imageIn.data){
        std::cout << "Error while opening file " << inputFileName << std::endl;
        return 0;
    }
    cv::cvtColor(imageIn, imageGray, cv::COLOR_BGR2GRAY);

    cv::Mat imageEdges;
    const double cannyThreshold1 = 100;
    const double cannyThreshold2 = 200;
    const int cannyAperture = 3;
    cv::Canny(imageGray, imageEdges, cannyThreshold1, cannyThreshold2, cannyAperture);

    int morphologySize = 2;
    cv::Mat edgesMorphed;
    cv::dilate(imageEdges, edgesMorphed, cv::Mat(), cv::Point(-1, -1), morphologySize);
    cv::erode(edgesMorphed, edgesMorphed, cv::Mat(), cv::Point(-1, -1), morphologySize);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(edgesMorphed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    cv::RNG rand(12345);
    std::vector<cv::RotatedRect> fittedEllipses(contours.size());
    for(int i = 0; i < contours.size(); i++){
        if(contours.at(i).size() > 5){
            fittedEllipses[i] = cv::fitEllipse(contours[i]);
        }
    }

    cv::Mat imageEllipse;
    imageIn.copyTo(imageEllipse);
    const int minEllipseInliers = 50;
    double total = 0;
    for(int i = 0; i < contours.size(); i++){
        if(contours.at(i).size() > minEllipseInliers){
            if(fittedEllipses[i].size.aspectRatio() < 0.95)
                continue;
            cv::Scalar color = cv::Scalar(rand.uniform(0, 256), rand.uniform(0,256), rand.uniform(0,256));
            cv::ellipse(imageEllipse, fittedEllipses[i], color, 5);
            total += value(fittedEllipses[i].size);
        }
    }
    cv::namedWindow("imageIn", cv::WINDOW_GUI_NORMAL);
    cv::imshow("imageIn", imageIn);
    cv::waitKey();

    cv::namedWindow("imageEllipse", cv::WINDOW_GUI_NORMAL);
    cv::imshow("imageEllipse", imageEllipse);
    cv::waitKey();

    for(auto it = change.begin(); it != change.end(); ++it) {
        std::cout << it->first << " - " << it->second << std::endl;
    }
    std::cout << "Total - " << total << std::endl;
}