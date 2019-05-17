#! /usr/bin/env python
from mrcnn_ros.srv import *

import rospy
import cv2
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError


def main():
    rospy.wait_for_service('instance_segment')
    instance_segment = rospy.ServiceProxy('instance_segment', InstanceSegmentation)

    # Get the test data path
    project_path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    project_path = os.path.abspath(project_path)
    test_data_path = os.path.join(project_path, 'test_data')
    cv_rbg_path = os.path.join(test_data_path, '000000_rgb.png')

    # Get the data
    cv_rgb = cv2.imread(cv_rbg_path, cv2.IMREAD_COLOR)

    # Build the request
    request = InstanceSegmentationRequest()
    bridge = CvBridge()
    request.rgb_image = bridge.cv2_to_imgmsg(cv_rgb, 'bgr8')

    # Call the service
    response = instance_segment(request)
    print(response.object_names)
    print(response.bounding_boxes)


if __name__ == '__main__':
    main()
