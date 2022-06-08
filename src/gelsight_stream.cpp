#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <map>
#include <string>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gelsight_stream");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
 
  std::string name; 
  int rate;
  bool publish_raw, publish_rectified, publish_mask; 
 
  ros::param::get("~rate", rate);
  ros::param::get("~publish_raw", publish_raw);
  ros::param::get("~publish_rectified", publish_rectified);
  ros::param::get("~publish_mask", publish_mask);

  cv::Point2f srcCorners[4];
  std::string corner_params[4] = { "~image_corners/top_left/", "~image_corners/top_right/",
	                           "~image_corners/bottom_left/", "~image_corners/bottom_right/" };
  for (int i = 0; i < 4; i++) {
    float x, y;
    ros::param::get(corner_params[i] + 'x', x);
    ros::param::get(corner_params[i] + 'y', y);
    srcCorners[i] = cv::Point2f(x, y);
  }

  float width, height;
  ros::param::get("~image_size/width", width);
  ros::param::get("~image_size/height", height);

  cv::Point2f dstCorners[4];
  dstCorners[0] = cv::Point2f(0.f,0.f);
  dstCorners[1] = cv::Point2f(width, 0.f);
  dstCorners[2] = cv::Point2f(0.f, height);
  dstCorners[3] = cv::Point2f(width, height);

  image_transport::Publisher raw_image_pub = it.advertise(cv::format("image/raw", name.c_str()), 1);
  image_transport::Publisher rect_image_pub = it.advertise(cv::format("image/rectified", name.c_str()), 1);
  image_transport::Publisher mask_image_pub = it.advertise(cv::format("image/mask", name.c_str()), 1);

  cv::VideoCapture capture;
  if (!capture.open(0, cv::CAP_ANY)) {
    ROS_ERROR("Failed to open Gelsight camera");
    return 1;
  } 
  
  if (capture.set(cv::CAP_PROP_FORMAT, CV_8UC1)) {
    ROS_WARN("Unable to set retrieval format");
  }
  
  cv::Mat raw_image;
  cv::Mat rect_image;
  cv::Mat grayscale_image;
  cv::Mat mask_image;
  
  ros::Rate loop_rate(rate);
  while (ros::ok())
  {
    if (capture.read(raw_image)) { 
      if (false) {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_image).toImageMsg();
        raw_image_pub.publish(msg);
      }

      cv::Mat image_tf = cv::getPerspectiveTransform(srcCorners, dstCorners);
      cv::warpPerspective(raw_image, rect_image, image_tf, cv::Size(width, height)); 

      if (publish_rectified) {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rect_image).toImageMsg();
        rect_image_pub.publish(msg);
      }

      if (publish_mask) {
        cv::cvtColor(rect_image, grayscale_image, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(grayscale_image, mask_image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 10);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask_image).toImageMsg();
        mask_image_pub.publish(msg);
      }
    } else {
      ROS_WARN("Frame dropped");
    }
    
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
