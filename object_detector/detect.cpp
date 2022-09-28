#include "detect.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>

Detect::Detect(std::string __modelPath){
    loadClassList(__modelPath);
    loadNet(__modelPath);
}

void Detect::loadClassList(std::string path){
    std::ifstream ifs(path + "game_classes.txt");
    std::string line;
    while(getline(ifs, line)){
        classList.push_back(line);
    }
    dimensions = 5 + classList.size();
}

void Detect::loadNet(std::string path){
    auto nn = cv::dnn::readNet(path + "game.onnx");
    nn.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    nn.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net = nn;
}

cv::Mat Detect::formatYOLOV5(const cv::Mat &source){
    int col = source.cols, row = source.rows, _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_MAKETYPE(CV_8U, source.channels()));
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    cv::cvtColor(result, result, cv::COLOR_BGRA2BGR);
    return result;
}

void Detect::detectObjects(cv::Mat &image){
    cv::Mat blob;
    auto inputImage = formatYOLOV5(image);
    
    cv::dnn::blobFromImage(
        inputImage, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
        cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float xFactor = inputImage.cols / INPUT_WIDTH;
    float yFactor = inputImage.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;
    const int rows = 25200;
    
    std::vector<int> classIDs;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];

        if (confidence >= CONFIDENCE_THRESHOLD) {
            float * scores = data + 5;
            cv::Mat scoreMatrix(1, classList.size(), CV_32FC1, scores);
            cv::Point classID;
            double score;
            cv::minMaxLoc(scoreMatrix, 0, &score, 0, &classID);

            if (score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                classIDs.push_back(classID.x);

                float x = data[0], y = data[1], w = data[2], h = data[3];
                int left = int((x - 0.5 * w) * xFactor), top = int((y - 0.5 * h) * yFactor);
                int width = int(w * xFactor), height = int(h * yFactor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        //  [x, y, w, h, conf, class_label1, ..., class_label26]
        data += dimensions;
    }

    std::vector<int> nms_result;
    objects.erase(objects.begin(), objects.end());
    tetherLines.erase(tetherLines.begin(), tetherLines.end());
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result(classIDs[idx], confidences[idx], boxes[idx]);
        objects[std::string(classList[result.classID])].push_back(result);
    }
    for(const auto& anchor : anchors){
        Detection result(-1, 1.0, cv::Rect(anchor.second, anchor.second));
        objects[anchor.first].push_back(result);
    }
}

void Detect::setTethers(std::vector<std::pair<std::string, std::string>> __tetherList){
    tetherList = __tetherList;
}

void Detect::addAnchor(std::string label, cv::Point point){
    anchors.push_back(std::pair<std::string, cv::Point>(label, point));
}

void Detect::tether(std::string classFrom, std::string classTo){
    const auto& from = objects[classFrom];
    const auto& to = objects[classTo];
    
    if(to.empty() || from.empty()){
        return;
    }

    auto color = colors[to.back().classID % colors.size()];
    for(const auto& fObj : from){
        for(const auto& tObj : to){
            cv::Rect fBox = fObj.box, tBox = tObj.box;
            cv::Point fPoint = fBox.tl() + cv::Point(
                static_cast<int>(fBox.width / 2), static_cast<int>(fBox.height / 2)
            );            
            cv::Point tPoint = tBox.tl() + cv::Point(
                static_cast<int>(tBox.width / 2), static_cast<int>(tBox.height / 2)
            );  
            tetherLines.push_back(Line(fPoint, tPoint, color));
        }
    }
}

void Detect::drawTethers(cv::Mat& frame){
    for(const auto& tether : tetherLines){
        cv::line(frame, tether.from, tether.to, tether.color, 2);
    }
}

void Detect::drawRects(cv::Mat& frame){
    for(const auto& [key, vec] : objects){
        for(const Detection& detection : vec){
            if(detection.classID == -1){
                continue;
            }
            cv::Rect box = detection.box;
            auto color = colors[detection.classID % colors.size()];
            
            cv::rectangle(frame, box, color, 3);
            cv::rectangle(
                frame, cv::Point(box.x, box.y - 20),
                cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(
                frame, classList[detection.classID].c_str(), cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
}

void Detect::feedImage(cv::Mat frame){
    detectObjects(frame);
    for(auto t : tetherList){
        tether(t.first, t.second);
    }
}

int Detect::runVideo(std::string videoName){
    cv::VideoCapture capture(videoName);

    if(!capture.isOpened()){
        std::cerr << "Error opening video file\n";
        return -1;
    }

    const int R_VAL = cv::cuda::getCudaEnabledDeviceCount();
    if(R_VAL == 0){
        std::cerr << "OpenCV is compiled without CUDA support\n";
        return R_VAL;
    } else if(R_VAL == -1){
        std::cerr << "CUDA driver is not installed, or is incompatible\n";
        return R_VAL;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    int frameCount = 0, totalFrames = 0;
    float fps = -1;

    cv::namedWindow("output", cv::WINDOW_NORMAL);
    while(true){
        capture.read(frame);
        if (frame.empty()){
            std::cout << "End of stream\n";
            break;
        }

        feedImage(frame);
        drawRects(frame);
        drawTethers(frame);

        frameCount++;
        totalFrames++;
        if(frameCount >= 30){
            auto end = std::chrono::high_resolution_clock::now();
            fps = frameCount * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>
                (end - start).count();
            frameCount = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if(fps > 0){
            std::ostringstream fpsLabel;
            fpsLabel << std::fixed << std::setprecision(2);
            fpsLabel << "FPS: " << fps;
            std::string fpsLabelStr = fpsLabel.str();
            cv::putText(
                frame, fpsLabelStr.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("output", frame);

        if(cv::waitKey(1) != -1){
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    std::cout << "Total frames: " << totalFrames << "\n";
    return 0;
}
