#include <iostream>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <thread>
#include <mutex>

#define NUM_COMNMAND_LINE_ARGUMENTS 1
#define DISPLAY_WINDOW_NAME "Video Frame"

#define WEST_LANE1 80
#define WEST_LANE2 275
#define WEST_LANE3 415
#define EAST_LANE1 650
#define EAST_LANE2 890
#define THIN 30

static int westbound_count = 0;
static int eastbound_count = 0;
static std::mutex mu;

static cv::Mat kernel(16, 16, CV_8U);

//  count cars and draw rects
void monitorLane(cv::Mat& processedFrame, cv::Mat lane, cv::Rect rectLane, bool dirWest){
    cv::dilate(lane, lane, kernel, cv::Point(-1, -1), 10);
    cv::erode(lane, lane, kernel, cv::Point(-1, -1), 5);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(lane, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> fittedRects(contours.size());
    for(int i = 0; i < contours.size(); i++){
        if(contours.at(i).size() > 1){
            fittedRects[i] = cv::boundingRect(contours[i]);
            if(fittedRects[i].size().height < 0.5*rectLane.height){
                continue;
            }
            fittedRects[i].height = rectLane.height + 2*THIN;
            fittedRects[i].y = rectLane.tl().y - THIN;
            mu.try_lock();
            cv::rectangle(
                processedFrame,
                fittedRects[i],
                dirWest ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                5
            );
            mu.unlock();
        }
    }
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

    int captureWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int captureHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int captureFPS = static_cast<int>(capture.get(cv::CAP_PROP_FPS));

    cv::namedWindow(DISPLAY_WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::namedWindow("fgMask", cv::WINDOW_NORMAL);

    bool doCapture = true;
    int frameCount = 0;
    const int bgHistory = 250;
    const float bgThreshold = 150;

    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(bgHistory, bgThreshold, false);
    cv::Mat fgMask;
    std::vector<cv::Rect> lanesWest;
    std::vector<cv::Rect> lanesEast;

    lanesWest.push_back(cv::Rect(cv::Point(0, 0), cv::Point(captureWidth, WEST_LANE1 - 2*THIN)));
    lanesWest.push_back(cv::Rect(cv::Point(0, WEST_LANE1 + THIN), cv::Point(captureWidth, WEST_LANE2 - THIN)));
    lanesWest.push_back(cv::Rect(cv::Point(0, WEST_LANE2 + THIN), cv::Point(captureWidth, WEST_LANE3 - THIN)));
    lanesEast.push_back(cv::Rect(cv::Point(0, WEST_LANE3 + THIN), cv::Point(captureWidth, EAST_LANE1 - THIN)));
    lanesEast.push_back(cv::Rect(cv::Point(0, EAST_LANE1 + THIN), cv::Point(captureWidth, EAST_LANE2 - THIN)));
    lanesEast.push_back(cv::Rect(cv::Point(0, EAST_LANE2 + THIN), cv::Point(captureWidth, captureHeight - THIN)));
    

    while(doCapture){
        double startTicks = static_cast<double>(cv::getTickCount());
        cv::Mat captureFrame;
        cv::Mat grayFrame;

        std::vector<std::thread> threads;

        bool captureSuccess = capture.read(captureFrame);
        if(captureSuccess){
            const int rangeMin = 0;
            const int rangeMax = 255;

            cv::cvtColor(captureFrame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::normalize(grayFrame, grayFrame, rangeMin, rangeMax, cv::NORM_MINMAX, CV_8UC1);
            pMOG2->apply(grayFrame, fgMask);

            for(const auto& lane : lanesWest){
                threads.push_back(std::thread([&](){
                    monitorLane(captureFrame, fgMask(lane), lane, true);
                }));
            }
            for(const auto& lane : lanesEast){
                threads.push_back(std::thread([&](){
                    monitorLane(captureFrame, fgMask(lane), lane, false);
                }));
            }
            for(auto& thread : threads){
                thread.join();
            }
            frameCount++;
        }
        else{
            break;
        }

        if(captureSuccess){
            cv::imshow(DISPLAY_WINDOW_NAME, captureFrame);
			cv::imshow("fgMask", fgMask);
            
            int delayMs = (1.0 / captureFPS) * 1000;
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            if(((char) cv::waitKey(1)) == 'q'){
                doCapture = false;
            }
        }
        double endTicks = static_cast<double>(cv::getTickCount());
        double elapsedTime = (endTicks - startTicks) / cv::getTickFrequency();
    }
    capture.release();
}