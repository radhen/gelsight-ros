#include "ros/ros.h"

#include <image_transport/image_transport.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gelsight_image_stream");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher stream_pub = it.advertise("/camera/color/image_raw", 1);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    ROS_INFO("%s", msg.data.c_str());

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}