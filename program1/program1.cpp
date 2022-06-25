#include <iostream>
#include "opencv2/opencv.hpp"
#include <chrono>

#define NUM_COMNMAND_LINE_ARGUMENTS 1

const static std::string WINDOW_NAME("ImageWindow");
static cv::Scalar eyeDropColor(255, 255, 255);
static cv::Point origin(0, 0);
static cv::Mat original;
static bool pencilDown = false;
std::chrono::time_point<std::chrono::system_clock> start, end;

typedef enum{
    EYEDROPPER,
    PENCIL,
    CROP,
    PAINTBUCKET
} mode;
mode currMode = EYEDROPPER;

void eyedropper(cv::Mat& mat, cv::Point point){
    eyeDropColor = mat.at<cv::Vec3b>(point);
    std::cout << "selected color: bgr(";
    std::cout << eyeDropColor[2] << ", ";
    std::cout << eyeDropColor[1] << ", ";
    std::cout << eyeDropColor[0] << ")\n";
}

void crop(cv::Mat& mat, cv::Point dest){
    cv::Rect roi(origin, dest);
    if(cv::Rect(origin, dest).size() == cv::Size(0,0)){
        return;
    }

    cv::Mat temp = mat(roi);
    if(temp.empty()){
        return;
    }
    temp.copyTo(mat);
    cv::imshow(WINDOW_NAME, mat);
}

void reset(cv::Mat& mat){
    original.copyTo(mat);
    cv::imshow(WINDOW_NAME, mat);
}

void pencil(cv::Mat& mat, cv::Point point){
    mat.at<cv::Vec3b>(point)[0] = eyeDropColor[0];
    mat.at<cv::Vec3b>(point)[1] = eyeDropColor[1];
    mat.at<cv::Vec3b>(point)[2] = eyeDropColor[2];
    cv::imshow(WINDOW_NAME, mat);
}

static void clickCallback(int event, int x, int y, int flags, void* param){
    cv::Mat& imageIn = *(cv::Mat*)param;  
    cv::Point point(x, y);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    start = std::chrono::system_clock::now();

    if(elapsed.count() < 1.0 && event == cv::EVENT_LBUTTONDBLCLK){
        reset(imageIn);
        return;
    }

    if(event == cv::EVENT_RBUTTONUP){
        switch(currMode){
            case EYEDROPPER:
                std::cout << "current tool: pencil\n";
                currMode = PENCIL;
                return;
            case PENCIL:
                std::cout << "current tool: crop\n";
                currMode = CROP;
                return;
            case CROP:
                std::cout << "current tool: paint bucket\n";
                currMode = PAINTBUCKET;
                return;
            case PAINTBUCKET:
                std::cout << "current tool: eye dropper\n";
                currMode = EYEDROPPER;
                return;
        }
    }

    switch(currMode){
        case EYEDROPPER:
            if(event == cv::EVENT_LBUTTONDOWN){
                eyedropper(imageIn, point);
            }
            return;
        case PENCIL:
            if(pencilDown){
                pencil(imageIn, point);
            }
            if(event == cv::EVENT_LBUTTONDOWN){
                pencilDown = true;
            } else if(event == cv::EVENT_LBUTTONUP){
                pencilDown = false;
            }
            return;
        case CROP:
            if(event == cv::EVENT_LBUTTONUP){
                crop(imageIn, point);
            } else if(event == cv::EVENT_LBUTTONDOWN){
                origin = point;
            }
            return;
        case PAINTBUCKET:
            if(event == cv::EVENT_LBUTTONUP){
                cv::Scalar ref = imageIn.at<cv::Vec3b>(point);
                cv::floodFill(imageIn, point, eyeDropColor);
                cv::imshow(WINDOW_NAME, imageIn);
            }
            return;
    }
}

int main(int argc, char **argv){
    std::string inputFileName;
    cv::Mat imageIn;

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
    imageIn.copyTo(original);
    start = std::chrono::system_clock::now();

    std::cout << "current tool: eye dropper\n";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_GUI_NORMAL);
    cv::imshow(WINDOW_NAME, imageIn);
    cv::setMouseCallback(WINDOW_NAME, clickCallback, &imageIn);
    cv::waitKey();
}