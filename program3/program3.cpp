#include <iostream>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <thread>
#include <mutex>

#define NUM_COMNMAND_LINE_ARGUMENTS 1
#define DISPLAY_WINDOW_NAME "Video Frame"

#define WEST_LANE1 75
#define WEST_LANE2 275
#define WEST_LANE3 415
#define EAST_LANE1 650
#define EAST_LANE2 890
#define MIDDLE 960

#define THIN 40
#define START_FRAME 10
#define THRESH_HEIGHT 0.4
#define MIN_AREA 8000
#define MAX_AREA 120000

static int westbound_count = 0;
static int eastbound_count = 0;
static std::mutex mu;
static cv::Mat kernel(16, 16, CV_8U);
static std::map<int, bool> isMiddle;

//  count cars and draw rects
void monitorLane(cv::Mat& processedFrame, cv::Mat lane, cv::Rect rectLane){
    bool dirWest = rectLane.br().y < WEST_LANE3;
    cv::dilate(lane, lane, kernel, cv::Point(-1, -1), 8);
    cv::erode(lane, lane, kernel, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(lane, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> fittedRects;
    //  filter and draw rectangles
    for(int i = 0; i < contours.size(); i++){
        if(contours.at(i).size() > 1){
            auto fittedRect = cv::boundingRect(contours[i]);
            if(fittedRect.size().height < THRESH_HEIGHT*rectLane.height){
                continue;
            }
            if(fittedRect.area() < MIN_AREA || fittedRect.area() > MAX_AREA){
                continue;
            }

            fittedRect.height = rectLane.height + 2*THIN;
            fittedRect.y = rectLane.tl().y - THIN;
            fittedRects.push_back(fittedRect);

            mu.try_lock();
            cv::rectangle(
                processedFrame,
                fittedRect,
                dirWest ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                5
            );
            mu.unlock();
        }
    }
    //  check if a car collides with middle and count
    for(const auto& rect : fittedRects){
        if(rect.tl().x <= MIDDLE && rect.br().x >= MIDDLE){
            if(!isMiddle[rectLane.br().y]){
                dirWest ? westbound_count++ : eastbound_count++;
                isMiddle[rectLane.br().y] = true;
            }
            return;
        }
    }
    isMiddle[rectLane.br().y] = false;
}

int main(int argc, char **argv){
    std::string fileName;
    if(argc != NUM_COMNMAND_LINE_ARGUMENTS + 1){
        std::printf("USAGE: %s <file_path> \n", argv[0]);
        return 0;
    }
    else{
        fileName = argv[1];
    }
    cv::VideoCapture capture(fileName);
    if(!capture.isOpened()){
        std::printf("Unable to open video source, terminating program! \n");
        return 0;
    }
    cv::namedWindow(DISPLAY_WINDOW_NAME, cv::WINDOW_NORMAL);

    const int captureWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int captureHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int captureFPS = static_cast<int>(capture.get(cv::CAP_PROP_FPS));

    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(250, 125, false);
    cv::Mat fgMask;
    std::vector<cv::Rect> lanes({   //  ROIs for lanes
        cv::Rect(cv::Point(0, 0), cv::Point(captureWidth, WEST_LANE1 - THIN)),
        cv::Rect(cv::Point(0, WEST_LANE1 + THIN), cv::Point(captureWidth, WEST_LANE2 - THIN)),
        cv::Rect(cv::Point(0, WEST_LANE2 + THIN), cv::Point(captureWidth, WEST_LANE3 - THIN)),
        cv::Rect(cv::Point(0, WEST_LANE3 + THIN), cv::Point(captureWidth, EAST_LANE1 - THIN)),
        cv::Rect(cv::Point(0, EAST_LANE1 + THIN), cv::Point(captureWidth, EAST_LANE2 - THIN)),
        cv::Rect(cv::Point(0, EAST_LANE2 + THIN), cv::Point(captureWidth, captureHeight - THIN))
    });

    for(const auto& lane : lanes){
        isMiddle[lane.br().y] = false;
    }
    
    int frameCount = 0;
    bool doCapture = true;
    while(doCapture){
        std::vector<std::thread> threads;
        cv::Mat captureFrame;
        cv::Mat grayFrame;

        bool captureSuccess = capture.read(captureFrame);
        if(!captureSuccess){
            break;
        }

        cv::cvtColor(captureFrame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::normalize(grayFrame, grayFrame, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        pMOG2->apply(grayFrame, fgMask);
        //  apply pMOG2 for a few frames for background before starting
        if(frameCount > START_FRAME){
            for(const auto& lane : lanes){
                threads.push_back(std::thread([&](){
                    monitorLane(captureFrame, fgMask(lane), lane);
                }));
            }
            for(auto& thread : threads){
                thread.join();
            }
        }
        frameCount++;

        if(captureSuccess){
            cv::imshow(DISPLAY_WINDOW_NAME, captureFrame);
            if(((char) cv::waitKey(1)) == 'q'){
                doCapture = false;
            }
        }

        std::cout << "WESTBOUND COUNT: " << westbound_count << "\n";
        std::cout << "EASTBOUND COUNT: " << eastbound_count << "\n\n";
    }
    capture.release();
}