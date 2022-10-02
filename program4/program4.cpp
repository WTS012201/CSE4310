#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/time.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

#define NUM_COMMAND_ARGS 1

using namespace std;

bool openCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut, const char* fileName)
{
    // convert the file name to string
    std::string fileNameStr(fileName);

    // handle various file types
    std::string fileExtension = fileNameStr.substr(fileNameStr.find_last_of(".") + 1);
    if(fileExtension.compare("pcd") == 0)
    {
        // attempt to open the file
        if(pcl::io::loadPCDFile<pcl::PointXYZRGBA>(fileNameStr, *cloudOut) == -1)
        {
            PCL_ERROR("error while attempting to read pcd file: %s \n", fileNameStr.c_str());
            return false;
        }
        else
        {
            return true;
        }
    }
    else if(fileExtension.compare("ply") == 0)
    {
        // attempt to open the file
        if(pcl::io::loadPLYFile<pcl::PointXYZRGBA>(fileNameStr, *cloudOut) == -1)
        {
            PCL_ERROR("error while attempting to read pcl file: %s \n", fileNameStr.c_str());
            return false;
        }
        else
        {
            return true;
        }
    }
    else
    {
        PCL_ERROR("error while attempting to read unsupported file: %s \n", fileNameStr.c_str());
        return false;
    }
}

bool saveCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloudIn, std::string fileName, bool binaryMode=true)
{
    // if the input cloud is empty, return
    if(cloudIn->points.size() == 0)
    {
        return false;
    }

    // attempt to save the file
    if(pcl::io::savePCDFile<pcl::PointXYZRGBA>(fileName, *cloudIn, binaryMode) == -1)
    {
        PCL_ERROR("error while attempting to save pcd file: %s \n", fileName);
        return false;
    }
    else
    {
        return true;
    }
}

bool segmentShape(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudIn,
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut,
                double distanceThreshold,
                int maxIterations,
                enum pcl::SacModel SACMODEL)
{
    // store the model coefficients
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    int r = 0, g = 0, b = 0;

    // Create the segmentation object for the shape model and set the parameters
    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SACMODEL);

    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);   
    seg.setRadiusLimits(0.1, 0.15);
    // Segment the largest shape component from the remaining cloud
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.setInputCloud(cloudIn);
    seg.segment(*inliers, *coefficients);

    //  return if no shape or bad size
    if(inliers->indices.size() < 800){
        return false;
    }
    //  set color based on shape info
    if(inliers->indices.size() > 30000 && SACMODEL == pcl::SACMODEL_PLANE){
        b = 255;
    } else if(SACMODEL == pcl::SACMODEL_PLANE){
        g = 255;
    } else if(SACMODEL == pcl::SACMODEL_SPHERE){
        r = 255;
    }

    for(int i = 0; i < inliers->indices.size(); i++){
        int index = inliers->indices.at(i);
        cloudIn->points.at(index).r = r;
        cloudIn->points.at(index).g = g;
        cloudIn->points.at(index).b = b;
    }
    pcl::ExtractIndices<pcl::PointXYZRGBA> filter;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

    //  move shape points from input cloud to output cloud
    filter.setInputCloud (cloudIn);
    filter.setIndices (inliers);
    filter.setNegative (false);
    filter.filter (*segCloud);
    filter.setNegative (true);
    filter.filter (*cloudIn);
    (*cloudOut) += (*segCloud);

    return true;
}

int main(int argc, char** argv){
    if(argc != NUM_COMMAND_ARGS + 1){
        std::printf("USAGE: %s <file_name>\n", argv[0]);
        return 0;
    }

    // parse the command line arguments
    char* fileName = argv[1];

    // create a stop watch for measuring time
    pcl::StopWatch watch;

    // start timing the processing step
    watch.reset();

    // open the point cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZRGBA>);
    openCloud(cloudIn, fileName);
    
    //  filter out points in the far back or very front of camera
    pcl::PassThrough<pcl::PointXYZRGBA> pass;
    pass.setInputCloud (cloudIn);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-1.0, -0.3);
    pass.filter(*cloudIn);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGBA>);

    int box_count = 0;
    int sphere_count = 0;
    
    //  segment shapes and count
    //  run until it cant detect any more shapes
    segmentShape(cloudIn, cloudOut, 0.0154, 5000, pcl::SACMODEL_PLANE);  //  get ground plane first
    while(segmentShape(cloudIn, cloudOut, 0.0020, 5000, pcl::SACMODEL_SPHERE)){ sphere_count++;}
    while(segmentShape(cloudIn, cloudOut, 0.0154, 5000, pcl::SACMODEL_PLANE)){ box_count++;}
    std::cout << "BOX COUNT: " << box_count << std::endl;
    std::cout << "SPHERE COUNT: " << sphere_count << std::endl;

    //  add remaining pixels
    (*cloudOut) += (*cloudIn);
    saveCloud(cloudOut, "output.pcd");
    // exit program
    return 0;
    
}
