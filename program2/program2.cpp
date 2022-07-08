#include <iostream>
#include "opencv2/opencv.hpp"

#define NUM_COMNMAND_LINE_ARGUMENTS 1
static std::map<std::string, int> change({
    {"Penny", 0},
    {"Nickel", 0},
    {"Dime", 0},
    {"Quarter", 0},
});

double value(cv::RotatedRect& fittedEllipse, cv::Mat& imageEllipse){
    cv::Size size = fittedEllipse.size;
    double approxDiam = (size.width + size.height) / 2;
    double value = 0;
    cv::Scalar color;
    
    if(approxDiam > 310 && approxDiam < 320){
        change["Penny"]++;
        color = cv::Scalar(0, 0, 255);
        value = 0.01;
    } else if(approxDiam > 345 && approxDiam < 370){
        change["Nickel"]++;
        color = cv::Scalar(0, 255, 255);
        value = 0.05;
    } else if(approxDiam > 290 && approxDiam < 300){
        change["Dime"]++;
        color = cv::Scalar(255, 0, 0);
        value = 0.10;
    } else if(approxDiam > 390 && approxDiam < 415){
        change["Quarter"]++;
        color = cv::Scalar(0, 255, 0);
        value = 0.25;
    } else{
        return value;
    }

    cv::ellipse(imageEllipse, fittedEllipse, color, 5);

    return value;
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
            total += value(fittedEllipses[i], imageEllipse);
        }
    }
    cv::namedWindow("imageIn", cv::WINDOW_GUI_NORMAL);
    cv::imshow("imageIn", imageIn);
    cv::waitKey();

    cv::namedWindow("imageEllipse", cv::WINDOW_GUI_NORMAL);
    cv::imshow("imageEllipse", imageEllipse);
    cv::waitKey();

    
    std::cout << "Penny - " << change["Penny"] << std::endl;
    std::cout << "Nickel - " << change["Nickel"] << std::endl;
    std::cout << "Dime - " << change["Dime"] << std::endl;
    std::cout << "Quarter - " << change["Quarter"] << std::endl;
    std::cout << "Total - $" << total << std::endl;
}