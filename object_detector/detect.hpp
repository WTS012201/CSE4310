#ifndef __DETECT_H
#define __DETECT_H

#include <fstream>
#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp" 

class Detect{
    private: 
        const std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
             cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)
        };
        cv::Point playerPoint;
        constexpr static float INPUT_WIDTH = 640.0;
        constexpr static float INPUT_HEIGHT = 640.0;
        constexpr static float SCORE_THRESHOLD = 0.2;
        constexpr static float NMS_THRESHOLD = 0.4;
        constexpr static float CONFIDENCE_THRESHOLD = 0.4;

        int dimensions;
        struct Line{
            cv::Point from;
            cv::Point to;
            cv::Scalar color;
        };
        // cv::Point side;
        std::vector<std::string> classList;
        cv::dnn::Net net;
        cv::Mat formatYOLOV5(const cv::Mat &source);
        std::vector<std::pair<std::string, std::string>> tetherList;
    public:
        struct Detection{
            int classID;
            float confidence;
            cv::Rect box;

            bool operator==(const Detection &rhs) const{
                return (classID == rhs.classID
                    && confidence == rhs.confidence
                    && box == rhs.box);
            }
        };
        typedef std::unordered_map<std::string, std::vector<Detection>> object_map_type;

        //  Object container for the current frame. Use object label as key
        object_map_type objects;
        //  Draw line in frame between objects
        std::vector<Line> tetherLines;
        //  set a point of reference to connect to other objects in the frame
        std::vector<std::pair<std::string, cv::Point>> anchors; 
        cv::Mat frame;

        Detect(std::string __modelPath);
        
        //  Loads the class label file
        void loadClassList(std::string fileName);
        //  Run the image classifier
        void detectObjects(cv::Mat& image);
        //  Create a line to be drawn between to objects
        void tether(std::string classFrom, std::string classTo);
        //  Draw tethers in frame
        void drawTethers(cv::Mat& frame);
        //  Draw rects in frame
        void drawRects(cv::Mat& frame);
        //  Load the network file
        void loadNet(std::string fileName);
        //  Set connections between objects in frame
        void setTethers(std::vector<std::pair<std::string, std::string>> tetherList);
        //  Feed a frame in for detection
        void feedImage(cv::Mat frame);
        //  Set a point(anchor) of reference in the frame
        void addAnchor(std::string label, cv::Point point);
        //  Run a sample video
        int runVideo(std::string videoName);
};
#endif