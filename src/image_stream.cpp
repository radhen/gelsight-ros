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
  ros::init(argc, argv, "gelsight_image_stream");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
 
  std::string name; 
  bool publish_raw, publish_rectified, publish_mask; 
 
  nh.getParam("name", name);
  nh.getParam("publish_raw", publish_raw);
  nh.getParam("publish_rectified", publish_rectified);
  nh.getParam("publish_mask", publish_mask);

  cv::Point2f srcCorners[4];
  std::map<std::string, double> top_left, top_right, bot_left, bot_right;
  nh.getParam("image_corners/top_left", top_left);
  srcCorners[0] = cv::Point2f((float) top_left["x"], (float) top_left["y"]);
  srcCorners[0] = cv::Point2f(158.f, 134.f);

  nh.getParam("image_corners/top_right", top_right);
  srcCorners[1] = cv::Point2f((float) top_right["x"], (float) top_right["y"]);
  srcCorners[1] = cv::Point2f(505.f, 109.f);

  nh.getParam("image_corners/bot_left", bot_left);
  srcCorners[2] = cv::Point2f((float) bot_left["x"], (float) bot_left["y"]);
  srcCorners[2] = cv::Point2f(163.f, 356.f);

  nh.getParam("image_corners/bot_right", bot_right);
  srcCorners[3] = cv::Point2f((float) bot_right["x"], (float) bot_right["y"]);
  srcCorners[3] = cv::Point2f(506.f, 386.f);

  double width, height;
  nh.getParam("image_size/width", width);
  nh.getParam("image_size/height", height);

  cv::Point2f dstCorners[4];
  dstCorners[0] = cv::Point2f(0.f,0.f);
  dstCorners[1] = cv::Point2f(640.f, 0.f);
  dstCorners[2] = cv::Point2f(0.f, 480.f);
  dstCorners[3] = cv::Point2f(640.f, 480.f);

  image_transport::Publisher raw_image_pub = it.advertise(cv::format("%s/image/raw", name.c_str()), 1);
  image_transport::Publisher rect_image_pub = it.advertise(cv::format("%s/image/rectified", name.c_str()), 1);
  image_transport::Publisher mask_image_pub = it.advertise(cv::format("%s/image/mask", name.c_str()), 1);

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
  
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    if (capture.read(raw_image)) { 
      if (false) {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_image).toImageMsg();
        raw_image_pub.publish(msg);
      }

      cv::Mat image_tf = cv::getPerspectiveTransform(srcCorners, dstCorners);
      cv::warpPerspective(raw_image, rect_image, image_tf, cv::Size(640.f, 480.f)); 

      if (publish_rectified) {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rect_image).toImageMsg();
        rect_image_pub.publish(msg);
      }

      continue;

      cv::cvtColor(rect_image, grayscale_image, cv::COLOR_BGR2GRAY);
      cv::adaptiveThreshold(grayscale_image, mask_image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 10);

      if (false) {
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
