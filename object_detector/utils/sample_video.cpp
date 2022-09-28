#include "../detect.hpp"

//  topside player pos cam lock(1035, 330)
//  botside player pos cam lock(865, 375)
int main(int argc, char **argv){
    if(argc != 3){
        std::cerr << "Usage: sample_video <model dir path> <video file path>";
        return -1;
    }
    cv::Point TOPSIDE = cv::Point(1035, 330);
    cv::Point BOTSIDE = cv::Point(865, 375);
    Detect detect = Detect(argv[1]);
    detect.addAnchor("player", TOPSIDE);

    std::vector<std::pair<std::string, std::string>> tetherList{
        std::pair<std::string, std::string>("ally", "enemy"),
        std::pair<std::string, std::string>("player", "enemy"),
        std::pair<std::string, std::string>("player", "ally"),
        std::pair<std::string, std::string>("player", "enemy_minion")
    };
    detect.setTethers(tetherList);
    detect.runVideo(argv[2]);
    return 0;
}