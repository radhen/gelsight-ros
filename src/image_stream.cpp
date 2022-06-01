#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <raspicam/raspicam_cv.h>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gelsight_image_stream");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher stream_pub = it.advertise("camera/color/image", 1);

  raspicam::RaspiCam_Cv camera;
  camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
  if (!camera.open()) {
    ROS_ERROR("Failed to open Gelsight camera");
    return 1;
  } 
  
  cv::Mat image;
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    camera.grab();
    camera.retrieve(image);

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    stream_pub.publish(msg);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}