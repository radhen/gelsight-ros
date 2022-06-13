#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from mars_msgs.srv import *


if __name__ == "__main__":
    rospy.wait_for_service('icp_mesh_tf')
    serv = rospy.ServiceProxy('icp_mesh_tf', ICPMeshTF)
    res = serv(ICPMeshTFRequest('large_round_peg'))
    print(res)
    